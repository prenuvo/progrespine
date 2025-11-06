from __future__ import annotations
from pathlib import Path, PosixPath
import argparse, csv, math, glob
import torch, lightning.pytorch as pl
import numpy as np

from progrespine.dataset.spine import make_loader
from progrespine.models import LitModelProto

# ---------- utils ----------
def to_floats(x):
    if x is None: return []
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().flatten().float().tolist()
    if isinstance(x, (list, tuple)):
        return to_floats(torch.as_tensor(x))
    try:
        return [float(x)]
    except Exception:
        return []

def summarize(preds, targets):
    if not targets or len(targets) != len(preds): return None, None
    diffs = [abs(p - t) for p, t in zip(preds, targets)]
    mae = sum(diffs) / len(diffs)
    rmse = (sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(preds)) ** 0.5
    return mae, rmse

def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["id", "pred", "target"])
        w.writerows(rows)

def pick_ckpt(ckpt_arg: Path) -> Path:
    """Allow passing a single .ckpt or a run/checkpoints directory."""
    if ckpt_arg.is_file() and ckpt_arg.suffix == ".ckpt":
        return ckpt_arg
    # treat as a directory: try to find a .ckpt inside
    candidates = sorted(map(Path, glob.glob(str(ckpt_arg / "*.ckpt"))))
    if not candidates:
        # maybe user pointed at the run root → look in run/checkpoints/
        candidates = sorted(map(Path, glob.glob(str(ckpt_arg / "checkpoints/*.ckpt"))))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt found under: {ckpt_arg}")
    # prefer most recent by mtime
    return max(candidates, key=lambda p: p.stat().st_mtime)

# ---------- loading ----------
def load_model(ckpt_path: Path, device: torch.device, dataloader_push):
    # Preferred path: Lightning
    try:
        model = LitModelProto.load_from_checkpoint(
            ckpt_path,
            map_location=device,
            strict=True,
            dataloader_push=dataloader_push,
        )
        model.eval().to(device)
        print(f"Loaded with load_from_checkpoint: {ckpt_path}")
        return model
    except Exception as e:
        print(f"[fallback to state_dict] {type(e).__name__}: {e}")

    # Fallback: raw state_dict (handles odd pickles/old runs)
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([PosixPath, np.dtype])
    except Exception:
        pass

    raw = torch.load(ckpt_path, map_location=device)
    state = raw.get("state_dict", raw)
    model = LitModelProto(dataloader_push=dataloader_push, prediction_r_init=5)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("state_dict loaded. Missing:", missing, "| Unexpected:", unexpected)
    model.eval().to(device)
    return model

# ---------- prediction ----------
def collect_outputs(raw_batches):
    preds, targets, ids = [], [], []
    for out in raw_batches:
        if isinstance(out, (tuple, list)):
            out = out[0]
            
        if isinstance(out, dict):
            p = out.get("pred")
            t = out.get("age")
            i = out.get("study_id")
            p_list = to_floats(p)
            t_list = to_floats(t) if t is not None else []
            if i is None:
                ids += [None] * len(p_list)
            elif isinstance(i, (list, tuple)):
                ids += [str(v) for v in i]
            else:
                ids += [str(i)] * len(p_list)
            preds += p_list
            targets += t_list
        else:
            preds += to_floats(out)
    if not ids or len(ids) != len(preds):
        ids = [f"sample_{k:05d}" for k in range(len(preds))]
    if len(targets) != len(preds):
        targets = targets + [None] * (len(preds) - len(targets))
    rows = []
    for i, p, t in zip(ids, preds, targets):
        rows.append([i, f"{p:.2f}", "" if t is None else f"{t:.2f}"])
    return rows

def run_predict(ckpt_path: Path, split: str, batch_size: int,
                data_root: Path, df_folder: Path, nifti_file: str, out_csv: Path):
    pl.seed_everything(42)

    # dataloader_push for the module
    dataloader_push = make_loader(
        "train", batch_size=2, shuffle=False,
        data_root=data_root, df_folder=df_folder, nifti_file=nifti_file
    )
    # inference loader
    loader = make_loader(
        split, batch_size=batch_size, shuffle=False,
        data_root=data_root, df_folder=df_folder, nifti_file=nifti_file
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_model(ckpt_path, device, dataloader_push)
    trainer = pl.Trainer(accelerator=accelerator, devices=1, logger=False)
    raw = trainer.predict(model, dataloaders=loader)

    rows = collect_outputs(raw)
    write_csv(out_csv, rows)

    preds_f = [float(r[1]) for r in rows if r[1] != ""]
    targs_f = [float(r[2]) for r in rows if r[2] != ""]
    mae, rmse = summarize(preds_f, targs_f)
    print(f"Saved {len(rows)} predictions → {out_csv}")
    if mae is not None:
        print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    for k, (i, p, t) in enumerate(rows[: min(5, len(rows))]):
        print(f"{k:02d} | id={i} | pred={p:>6} | target={(t if t else 'NA'):>6}")

# ---------- CLI ----------
def parse_args():
    home = Path.home()
    p = argparse.ArgumentParser("Run inference with a Lightning .ckpt")
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to .ckpt OR to a run or checkpoints directory.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--data-root", type=Path, default=home / "data_age" / "spine")
    p.add_argument("--df-folder", type=Path, default=home / "data_age" / "meta" / "spine")
    p.add_argument("--nifti-file", type=str, default="t2_whole_spine_masked_resampled.nii.gz")
    p.add_argument("--out-csv", type=Path, default=Path("predictions.csv"))
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ckpt_path = pick_ckpt(args.ckpt)
    run_predict(ckpt_path, args.split, args.batch_size,
                args.data_root, args.df_folder, args.nifti_file, args.out_csv)
