from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

import numpy as np

from tqdm import tqdm
import time
import lightning.pytorch as pl
from pathlib import Path


class RegressionMetric(nn.Module):
    def __init__(self, sigma: float = 1.0, w_leak: float = 0.2, epsilon: float = 1e-9, num_repeat: int = 3):
        super().__init__()
        self.sigma = sigma
        self.w_leak = w_leak
        self.epsilon = epsilon
        self.num_repeat = num_repeat

    def forward(
        self,
        distances: torch.Tensor,         # [B, P * R]
        labels: torch.Tensor,            # [B]
        proto_labels: torch.Tensor       # [P * R]
    ) -> torch.Tensor:

        B = distances.shape[0]
        P_total = proto_labels.shape[0]
        assert P_total % self.num_repeat == 0, "Total number of prototypes must be divisible by num_repeat"
        P = P_total // self.num_repeat

        # Step 1: reshape distances → [B, P, R]
        distances = distances.view(B, P, self.num_repeat)

        # Step 2: softmin over R (repeated protos per class)
        softmin_weights = F.softmin(distances, dim=2)
        aggregated_distances = torch.sum(softmin_weights * distances, dim=2)  # → [B, P]

        # Step 3: collapse proto_labels to one per class → [P]
        proto_labels = proto_labels.view(P, self.num_repeat)[:, 0]  # take first per group (assumed identical)

        # Step 4: compute label-prototype label differences
        labels = labels.view(-1, 1)             # [B, 1]
        proto_labels = proto_labels.view(1, -1) # [1, P]
        label_diff = torch.abs(labels - proto_labels)  # [B, P]

        # Step 5: compute Gaussian weight over label similarity
        weights = torch.exp(-((label_diff / self.sigma) ** 2) / 2) + self.w_leak
        weights = weights.detach()  # prevent gradients flowing through weights

        # Step 6: compute loss between distance and semantic label alignment
        diff = torch.abs(aggregated_distances - label_diff)  # [B, P]
        loss = torch.sum(weights * diff) / (weights.sum() + self.epsilon)
        return loss


class ProtoConfig:
    def __init__(self, img_size=[1, 14, 793, 384], proto_range=(25, 85, 2), num_repeat=3):
        self.img_size = img_size
        self.num_repeat = num_repeat

        # Ensure 84 is included
        proto_classes = list(range(*proto_range))
        proto_classes.append(84)

        proto_classes_tensor = torch.tensor(proto_classes)

        self.proto_classes = proto_classes_tensor.repeat(num_repeat).sort()[0]
        self.num_prototypes = self.proto_classes.numel()

        # Feature extractor and derived shape
        self.features = RegressionProto()
        dummy_data = torch.zeros((1, *img_size))
        features = self.features(dummy_data)
        self.output_size_conv = features.shape[1:]
        self.proto_shape = [self.num_prototypes] + list(self.output_size_conv)


class PushPrototypes:
    def __init__(self, pl_model: pl.LightningModule, config: ProtoConfig):
        self.ppnet = pl_model.ppnet
        self.ppnet.eval()
        self.device = pl_model.device
        self.current_epoch = pl_model.current_epoch

        self.proto_classes = config.proto_classes
        self.num_prototypes = config.num_prototypes
        self.proto_shape = config.proto_shape
        self.img_size = config.img_size
        self.num_repeat = config.num_repeat
        self.num_channels = config.output_size_conv[0]
        self.proto_study_ids = [""] * self.num_prototypes

    def push_prototypes(self, dataloader: torch.utils.data.DataLoader[dict[str, torch.Tensor]]) -> None:
        self.init_save_variables()

        for batch in tqdm(dataloader):
            image = batch["t2_whole_spine"]["mask"]
            study_id = batch["study_id"]
            self.update_protos_batch(image.to(self.device), study_id)

        self.update_model_variables()

    def init_save_variables(self) -> None:
        self.global_mindist = np.full(self.num_prototypes, np.inf)
        self.proto_latent_repr = np.zeros(tuple(self.proto_shape), dtype=np.float32)

    def update_protos_batch(self, batch_im: torch.Tensor, batch_study_id: list[str]) -> None:
        """
        batch_im: B x C x D x H x W
        """
        with torch.no_grad():
            batch_convfeatures = self.ppnet.features(batch_im) # B x FC x FD x FH x FW
            batch_protodist = self.ppnet.prototype_distances_ot(batch_im) # B x P

        batch_convfeatures = batch_convfeatures.detach().cpu().numpy()
        batch_protodist = batch_protodist.detach().cpu().numpy()

        proto_first = batch_protodist.swapaxes(0, 1) # P x B
        dist_perproto = proto_first.reshape(self.num_prototypes, -1) # P x B

        # Minimum distances and their locations per prototype
        mindist = np.amin(dist_perproto, axis=1)
        mindist_index = np.argmin(dist_perproto, axis=1)

        # Consistency check for first prototype
        assert proto_first[0, mindist_index[0]] == mindist[0]

        for proto_j in range(self.num_prototypes):
            if self.global_mindist[proto_j] > (newdist := mindist[proto_j]):
                bi = mindist_index[proto_j]
                self.global_mindist[proto_j] = newdist    
                self.proto_latent_repr[proto_j] = batch_convfeatures[bi]
                self.proto_study_ids[proto_j] = batch_study_id[bi]

    def update_model_variables(self) -> None:
        """
            update the model with the new prototype representations
        """
        ### FIXME: Roozbeh: Sus, not using grads and also the way we copy
        with torch.no_grad():
            self.ppnet.prototype_vectors.data.copy_(
                torch.tensor(self.proto_latent_repr, dtype=torch.float32, device=self.ppnet.prototype_vectors.device)
            )
        # self.ppnet.prototype_images = torch.tensor(self.prototype_images)
        self.ppnet.proto_study_ids = self.proto_study_ids


def batched_optimal_transport_log(
    dist_matrix: torch.Tensor,
    weights_a: torch.Tensor,
    weights_b: torch.Tensor,
    reg: float = 0.01,
    threshold: float = 0.01,
    max_iter: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes batched optimal transport in log space using Sinkhorn iterations."""

    assert weights_a.shape == weights_b.shape
    assert len(weights_a.shape) in {1, 3}

    if len(weights_a.shape) == 3:
        mu, nu = weights_a, weights_b
    else:
        mu = weights_a.view(1, 1, -1)
        nu = weights_b.view(1, 1, -1)

    u, v = torch.zeros_like(mu), torch.zeros_like(nu)
    logmu, lognu = torch.log(mu), torch.log(nu)
    K = -dist_matrix / reg

    for _ in range(max_iter):
        u_prev = u
        u = logmu - torch.logsumexp(K + v[:, :, None, :], dim=3)
        v = lognu - torch.logsumexp(K + u[:, :, :, None], dim=2)
        if (u - u_prev).abs().max() < threshold:
            break

    T = torch.exp(K + u[:, :, :, None] + v[:, :, None, :])
    return T, (u - u_prev).abs().max()


class RegressionProto(nn.Module):
    def __init__(
    self,
    in_ch: int = 1,
    depths: tuple[int, int, int, int, int, int] = (32, 64, 128, 256, 256, 64),
    ):
        super().__init__()

        torch.use_deterministic_algorithms(False)

        self.block1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_ch, depths[0], 3, padding=1)),
            ('norm', nn.BatchNorm3d(depths[0], momentum=0.01, eps=0.001)),
            ('relu', nn.ReLU()),
            ('pool', nn.MaxPool3d(kernel_size=(2, 2, 2))),  # 16×768×256 → 8×384×128
        ]))
        self.block2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(depths[0], depths[1], 3, padding=1)),
            ('norm', nn.BatchNorm3d(depths[1], momentum=0.01, eps=0.001)),
            ('relu', nn.ReLU()),
            ('pool', nn.MaxPool3d(kernel_size=(2, 2, 2))),  # → 4×192×64
        ]))
        self.block3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(depths[1], depths[2], 3, padding=1)),
            ('norm', nn.BatchNorm3d(depths[2], momentum=0.01, eps=0.001)),
            ('relu', nn.ReLU()),
            ('pool', nn.MaxPool3d(kernel_size=(2, 2, 2))),  # → 2×96×32
        ]))
        self.block4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(depths[2], depths[3], 3, padding=1)),
            ('norm', nn.BatchNorm3d(depths[3], momentum=0.01, eps=0.001)),
            ('relu', nn.ReLU()),
            ('pool', nn.MaxPool3d(kernel_size=(1, 2, 2))),  # → 1×48×16
        ]))
        self.block5 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(depths[3], depths[4], 3, padding=1)),
            ('norm', nn.BatchNorm3d(depths[4], momentum=0.01, eps=0.001)),
            ('relu', nn.ReLU()),
            ('pool', nn.MaxPool3d(kernel_size=(1, 2, 2))),  # → 1×24×8
        ]))
        self.top = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(depths[4], depths[5], kernel_size=(1, 1, 1), padding='same')),  # 1×24×8 → 1×24×12
            ('norm', nn.BatchNorm3d(depths[5], momentum=0.01, eps=0.001)),
            ('relu', nn.Sigmoid()),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.top(x)
        return x


class PPNetWholeProtos(nn.Module):
    def __init__(self, prediction_r_init, config: ProtoConfig):
        super().__init__()
        self.config = config

        self.img_size = config.img_size
        proto_classes = config.proto_classes
        self.register_buffer("proto_classes", proto_classes.float())

        self.num_prototypes = config.num_prototypes
        self.output_size_conv = config.output_size_conv
        self.proto_shape = config.proto_shape
        self.num_repeat = config.num_repeat
        self.num_channels = self.output_size_conv[0]

        self.features = config.features
        self.prototype_vectors = nn.Parameter(torch.rand(self.proto_shape), requires_grad=True)
        self.ones = nn.Parameter(torch.ones(self.proto_shape), requires_grad=False)
        self.r = nn.Parameter(torch.tensor([prediction_r_init]), requires_grad=False)
        self.embedding_loss_s = nn.Parameter(torch.ones(1) * 10.0, requires_grad=True)
        self.num_patches = self.output_size_conv[0] * self.output_size_conv[1] * self.output_size_conv[2]
        self.proto_study_ids = [""] * self.num_prototypes


    def get_initial_distributions(self, m: int) -> tuple[torch.Tensor, torch.Tensor]:
        dist_x = torch.full((m,), 1 / m, device=self.prototype_vectors.device)
        return dist_x, dist_x

    def compute_OT_distances(
        self,
        conv_features: torch.Tensor,
        prototypes_res: torch.Tensor,
        shape1: tuple[int, int],
        shape2: tuple[int, int],
    ) -> torch.Tensor:
        features = conv_features.permute(0, 2, 3, 4, 1).reshape(-1, conv_features.shape[1])
        patch_distances = torch.cdist(features, prototypes_res, p=2)
        patch_distances = patch_distances.view(shape1[0], shape1[1], shape2[0], shape2[1])
        patch_distances_res = torch.einsum("bipj->bpij", patch_distances)

        m = patch_distances_res.shape[-2]
        image_dist, proto_dist = self.get_initial_distributions(m)

        OT, _ = batched_optimal_transport_log(patch_distances_res, image_dist, proto_dist, reg=0.1)
        distances = torch.sum(OT * patch_distances_res, dim=(2, 3)) * self.embedding_loss_s
        return distances # B x P

    def prototype_distances_ot(self, x: torch.Tensor) -> torch.Tensor:
        conv_features = self.features(x) # B x FC x FD x FH x FW
        features = conv_features.permute(0, 2, 3, 4, 1)  # [B, FD, FH, FW, FC]
        prototypes = self.prototype_vectors.permute(0, 2, 3, 4, 1)  # [P, FD, FH, FW, FC]
        prototypes_res = prototypes.reshape(-1, prototypes.shape[-1])  # [P*FD*FH*FW, FC]

        message = f"Mismatch between prototype ({prototypes.shape}) and feature ({features.shape}) shapes"
        assert prototypes.shape[1:] == features.shape[1:], message

        distances = self.compute_OT_distances(
            conv_features,
            prototypes_res,
            (features.shape[0], features.shape[1] * features.shape[2] * features.shape[3]),
            (prototypes.shape[0], prototypes.shape[1] * prototypes.shape[2] * prototypes.shape[3]),
        )

        return distances # B x P

    def compute_age_predictions(self, min_distances: torch.Tensor) -> torch.Tensor:
        clipped = torch.where(min_distances > self.r, torch.inf, min_distances)
        weights = torch.exp(-clipped**2 / (2 * (self.r.to(min_distances.device) / 3)**2))
        
        proto_labels = self.proto_classes[None, :].to(weights.device)

        sum_w = torch.sum(weights, dim=1)
        eps = 1e-6  # or 1e-8 for finer tolerance
        prediction_raw = torch.sum(proto_labels * weights, dim=1) / (sum_w + eps)

        numnans = torch.isnan(prediction_raw).sum()

        if numnans > 0:
            sum_w = torch.sum(weights, dim=1)
            closest_idx = torch.argmin(min_distances, dim=1)
            fallback = self.proto_classes[closest_idx]
            prediction = torch.where(sum_w == 0, fallback, prediction_raw)

            for i in range(sum_w.shape[0]):
                if sum_w[i] == 0:
                    weights[i, closest_idx[i]] = 1
        else:
            prediction = prediction_raw
        
        return prediction

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        min_distances = self.prototype_distances_ot(x)
        logits = self.compute_age_predictions(min_distances)
        return logits, min_distances


class LitModelProto(pl.LightningModule):
    def __init__(self, dataloader_push, prediction_r_init=15) -> None:
        super().__init__()

        self.optimizer_steps_this_epoch = 0
        
        self.prediction_r_init = prediction_r_init
        self.proto_config=ProtoConfig()
        self.ppnet = PPNetWholeProtos(prediction_r_init=self.prediction_r_init, config=self.proto_config)
        self.protoloss = RegressionMetric(sigma=1.0, w_leak=0.05)
        self.accumulate_grad_batches = 3

        self.metric_function = torchmetrics.MeanAbsoluteError()
        self.validation_outputs = []

        self.dataloader_push = dataloader_push
        self.push_epoch = False
        self.push_history = []
        self.push_epochs = [4, 9, 14, 19, 24, 29, 34, 39, 44, 49]

        print("Proto labels:", self.ppnet.proto_classes.cpu().numpy())

    
    @property
    def automatic_optimization(self):
        return False

    @automatic_optimization.setter
    def automatic_optimization(self, value):
        raise AttributeError("Prototype learning does not support 'automatic_optimization'.")

    def get_logging_callback(self):
        """
        You can define any number of model-specific hooks like this -- must end with "_callback"
        They’ll be automatically discovered and registered. 
        """
        print(f"[INFO] Calling model callback method: get_logging_callback()")
        return LogPredictionsCallback(dataloader=self.dataloader_push)

    def forward(self, batch: dict[str, torch.Tensor], batch_idx: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        images = batch["t2_whole_spine"]["mask"]
        logits, min_distances = self.ppnet(images)
        return logits, min_distances

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        start = time.time()

        _, min_distances = self.ppnet(batch["t2_whole_spine"]["mask"])

        proto_labels = self.ppnet.proto_classes
        loss = self.protoloss(min_distances, batch["age"], proto_labels)
        self.log("train/loss", loss)

        optimizer = self.optimizers()  # type:ignore
        scheduler = self.lr_schedulers()
        
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            self.optimizer_steps_this_epoch += 1 
        
        if self.trainer.is_last_batch:
            scheduler.step()

        #logging gradient of the prototype_vectors
        if self.ppnet.prototype_vectors.grad is not None:
            grad_norm = self.ppnet.prototype_vectors.grad.norm().item()
            self.logger.experiment.log_metric(
                key="train/proto_vector_grad_norm",
                value=grad_norm,
                step=self.global_step,
                run_id=self.logger.run_id,
            )

        # Logging gradient norm of last conv layer in the backbone
        last_conv = self.ppnet.features.top[0]  # This is the final Conv3d layer
        if last_conv.weight.grad is not None:
            backbone_grad_norm = last_conv.weight.grad.norm().item()
            self.logger.experiment.log_metric(
                key="train/backbone_final_conv_grad_norm",
                value=backbone_grad_norm,
                step=self.global_step,
                run_id=self.logger.run_id,
            )

        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        logits, _ = self.ppnet(batch["t2_whole_spine"]["mask"])  # first element is age preds

        out = {"pred": logits}  # shape [B]

        if "age" in batch:
            out["age"] = batch["age"]  # shape [B]

        if "study_id" in batch:
            out["study_id"] = batch["study_id"]  # list/array of length B

        return out
    
    def on_train_epoch_end(self):
        if self.optimizer_steps_this_epoch == 0:
            print(f"[WARNING] No optimizer step occurred in epoch {self.current_epoch}. "
                f"Model will not learn. Check accumulate_grad_batches or dataset size.")
        else:
            print(f"[INFO] Optimizer stepped {self.optimizer_steps_this_epoch} time(s) in epoch {self.current_epoch}.")

        self.optimizer_steps_this_epoch = 0  # reset for next epoch


    def validation_step(self, batch, batch_idx):    
        """
        Logging loss data point per validation batch.
        """
        logits, min_distances = self.ppnet(batch["t2_whole_spine"]["mask"])
        proto_labels = self.ppnet.proto_classes
        loss = self.protoloss(min_distances, batch["age"], proto_labels)
        self.metric_function(logits, batch["age"])
        self.log("validation/loss", loss,  on_step=True, on_epoch=True)#Lightning logs two values: validation/loss_step and validation/loss_epoch
        self.validation_outputs.append({"loss": loss.detach()})
        return {"loss": loss} 

    def configure_optimizers(self): # type: ignore
        specs = [
            {"params": self.ppnet.features.parameters()},
            {"params": self.ppnet.prototype_vectors},
            {"params": self.ppnet.embedding_loss_s, "lr": 0.01}
        ]
        optimizer = torch.optim.Adam(specs, lr=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [{"scheduler": scheduler, "name": "joint_scheduler"}]

    def on_validation_epoch_end(self):
        """
        make use of the returned {"loss": ...} and log your MAE. 
        Logging loss: one data point per epoch
        Note: No outputs argument (Lightning v2 compliant)
        """
        # Log average loss over all validation batches
        # Compute average validation loss manually
        avg_loss = torch.stack([x["loss"] for x in self.validation_outputs]).mean()
        self.log("validation/avg_loss", avg_loss, on_epoch=True, prog_bar=True)

        # Compute and log metric
        mae = self.metric_function.compute()
        self.log("validation/mean_absolute_error", mae, on_epoch=True, prog_bar=True)
        self.metric_function.reset()

        # Clear accumulated outputs
        self.validation_outputs.clear()

    def on_validation_epoch_start(self) -> None:
        self.epoch_start_time = time.time()

        # Print proto labels (for inspection)
        proto_labels = self.ppnet.proto_classes.detach().cpu().numpy()
        self.log("embedding_loss_s", self.ppnet.embedding_loss_s, on_epoch=True)

        # Push prototypes if this epoch is listed
        current_epoch = self.current_epoch
        if current_epoch in self.push_epochs and current_epoch not in self.push_history:
            print(f"[INFO] Pushing prototypes at epoch {current_epoch}")
            self.ppnet.proto_classes = self.ppnet.proto_classes.to(self.device)
            pusher = PushPrototypes(self, self.proto_config)
            pusher.push_prototypes(self.dataloader_push)
            self.push_history.append(current_epoch)

            artifact_file = f"prototypes_epoch_{current_epoch}.txt"
            message = f"[INFO] Prototypes pushed (epoch={current_epoch}):"
            self.logger.experiment.log_text(
                text=message + "\n" + "\n".join(self.ppnet.proto_study_ids),
                artifact_file=artifact_file,
                run_id=self.logger.run_id,
            )
            print(f"[MLFLOW] Logged prototype study_ids for epoch {current_epoch}")


    def on_fit_start(self) -> None:
        print("[INFO] Push: on_fit_start: prototype pushing. ")

        if 0 in self.push_history:
            print("[DEBUG] Prototypes already pushed at epoch 0.")
            return

        assert self.ppnet.proto_classes is not None, "[ERROR] proto_classes is None before pushing prototypes."
        #self.ppnet.proto_classes = self.ppnet.proto_classes.to(self.device)
        self.ppnet = self.ppnet.to(self.device)


class LogPredictionsCallback(pl.Callback):
    def __init__(self, dataloader, max_batches=3):
        self._predictions = []
        self._targets = []
        self.dataloader = dataloader
        self.max_batches = max_batches

    def _reset_epoch_storage(self):
        """
            Reset predictions and targets each epoch to avoid accumulation
        """
        self._predictions = []
        self._targets = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self._reset_epoch_storage()
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        x = batch["t2_whole_spine"]["mask"]
        with torch.no_grad():
            pred, _ = pl_module.ppnet(x)
        self._predictions.extend(pred.cpu().tolist())
        self._targets.extend(batch["age"].cpu().tolist())

    def _generate_lines(self, preds, targets):
        return [
            f"Sample {i}: GT = {gt:.2f}, Pred = {pred:.2f}, Error = {abs(gt - pred):.2f}"
            for i, (gt, pred) in enumerate(zip(targets, preds))
        ]

    def _log_lines_to_file(self, lines, filename, logger):
        try:
            local_dir = Path("mlruns") / 'last_run_predicted_age'
            local_dir.mkdir(parents=True, exist_ok=True)
            text_path = local_dir / filename
            text_path.write_text("\n".join(lines))
            print(f"[DEBUG] File written to: {text_path}")

            logger.experiment.log_artifact(
                run_id=logger.run_id,
                local_path=str(text_path),
                artifact_path="predicted_age"
            )
            print(f"[DEBUG] log_artifact completed for {filename}")

        except Exception as e:
            print(f"[ERROR] Logging {filename} failed:", e)

    def on_fit_start(self, trainer, pl_module):
        print("[INFO] Logging predictions before training (on_fit_start)")

        predictions, targets = [], []
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= self.max_batches:
                    break
                x = batch["t2_whole_spine"]["mask"].to(pl_module.device)
                y = batch["age"].to(pl_module.device)
                preds, _ = pl_module.ppnet(x)
                predictions.extend(preds.cpu().tolist())
                targets.extend(y.cpu().tolist())

        lines = self._generate_lines(predictions, targets)
        self._log_lines_to_file(lines, "pretrain.txt", trainer.logger)

    def on_validation_epoch_end(self, trainer, pl_module):
        filename = f"val_predictions_epoch_{trainer.current_epoch}.txt"
        print(f"[INFO] Logging predictions for epoch {trainer.current_epoch}")
        lines = self._generate_lines(self._predictions, self._targets)
        self._log_lines_to_file(lines, filename, trainer.logger)
            