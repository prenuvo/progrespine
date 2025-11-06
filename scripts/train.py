from pathlib import Path
import torch, lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader

from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from progrespine.models import LitModelProto
from progrespine.dataset.spine import make_loader

def train():
    pl.seed_everything(42)

    data_root = Path.home() / 'data_age' / 'spine'

    df_folder = Path.home() / 'data_age' / 'meta' / 'spine'
    nifti_file = 't2_whole_spine_masked_resampled.nii.gz'


    # Dummy "push" loader 
    dataloader_push = make_loader('train', batch_size=2, shuffle=False, data_root=data_root, df_folder=df_folder, nifti_file=nifti_file)

    # Instantiate LightningModule
    model = LitModelProto(dataloader_push=dataloader_push, prediction_r_init=5)

    train_loader = make_loader('train', batch_size=2, shuffle=True, data_root=data_root, df_folder=df_folder, nifti_file=nifti_file)
    val_loader   = make_loader('val', batch_size=2, shuffle=False, data_root=data_root, df_folder=df_folder, nifti_file=nifti_file)

    # MLflow logger
    logger = MLFlowLogger(
        experiment_name="proto-regression",
        tracking_uri="file:./mlruns"   # local folder storage
    )

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(monitor=None, save_top_k=0, save_last=True),
        model.get_logging_callback(),  # LogPredictionsCallback using dataloader_push
    ]

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=1.0,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    train()