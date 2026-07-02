"""
train.py
--------
Usage:
    python train.py --config configs/unet.yaml
    python train.py --config configs/unet.yaml --device cuda:1
    python train.py --config configs/unet.yaml --resume logs/.../fold_1/checkpoints/best_model.pth

    # CSV-based splits (five folds)
    python train.py --config configs/unet.yaml \\
                    --set dataset.split_csv_dir=splits/split_train_val_test

    # Single CSV (single fold)
    python train.py --config configs/unet.yaml \\
                    --set dataset.split_csv=splits/split_train_val_test/split_train_val_test_mydata_fold1.csv

    # Override any yaml parameter at runtime
    python train.py --config configs/unet.yaml \\
                    --set dataset.images_dir=/new/path \\
                          training.epochs=50 \\
                          training.batch_size=4 \\
                          model.n_classes=3
Seed behaviour
--------------
  All modes except training_only:
      Every fold uses the same base seed (training.seed, default 42).
      Data already differs between folds so seed variation is unnecessary.
      
  training_only mode:
      All folds see identical data (full dataset), so each fold gets a
      different seed to vary weight init, augmentation, dropout, and
      DataLoader shuffle order:
          fold 1  ->  base_seed + 1  (e.g. 43)
          fold 2  ->  base_seed + 2  (e.g. 44)
          ...
      Set training.seed in your config to control the base value.
"""

import argparse
import csv
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch
import yaml

from utils.config      import load_config, print_config
from utils.dataset     import get_splits
from utils.train_utils import ModelProcess
from modules           import get_model

# ── Seed ───────────────────────────────────────────────────────────────────────
def _set_seed(seed: int):
    """Set all relevant random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _fold_seed(base_seed: int, fold_num: int, eval_mode: str, seed_list: list = None) -> int:
    """
    Return the seed to use for a given fold.

    training_only : all folds have identical data, so each fold gets a
                    unique seed to vary weight init, augmentation, dropout,
                    and DataLoader shuffle order.
                    Seeds are taken from seed_list in order:
                      fold 1 -> seed_list[0], fold 2 -> seed_list[1], ...
                    If there are more folds than seeds, cycles back to
                    seed_list[0], seed_list[1], ... (modulo wrap).
                    If seed_list is None/empty, falls back to base_seed + fold_num.
    all other modes: every fold uses base_seed — data already differs
                     between folds.
    """
    if eval_mode == "training_only":
        if seed_list:
            return seed_list[(fold_num - 1) % len(seed_list)]
        return base_seed + fold_num
    return base_seed

# ── Args ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Semantic Segmentation Training")
    p.add_argument("--config",  type=str, required=True, help="Path to YAML config (e.g. configs/unet.yaml)")
    p.add_argument("--device",  type=str, default=None, help="Device: cuda, cuda:0, cpu (default: auto-detect)")
    p.add_argument("--resume",  type=str, default=None, help="Checkpoint path to resume from (single-fold only)")
    p.add_argument("--set",     nargs="*", default=[], metavar="KEY=VALUE", help="Override config params e.g. --set dataset.images_dir=/path training.epochs=50  dataset.split_csv_dir=splits/split_train_val_test")
    return p.parse_args()

# ── Config override ────────────────────────────────────────────────────────────
def _apply_overrides(cfg: dict, overrides: list) -> dict:
    """
    Apply CLI overrides to a config dict (dot-notation, auto-cast).
    Examples:
        dataset.images_dir=/data/patches
        training.epochs=100
        training.amp=false
        dataset.split_csv_dir=splits/split_train_val_test
        dataset.split_csv=splits/split_train_val_test/split_train_val_test_ds_fold1.csv
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set override must be KEY=VALUE, got: '{item}'")
        key_path, _, value = item.partition("=")
        keys = key_path.strip().split(".")

        node = cfg
        for k in keys[:-1]:
            if k not in node:
                raise KeyError(f"Config key '{k}' not found in path '{key_path}'")
            node = node[k]

        final_key = keys[-1]
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        else:
            try:    value = int(value)
            except ValueError:
                try: value = float(value)
                except ValueError:
                    pass   # keep as string

        node[final_key] = value
        print(f"  Config override: {key_path} = {value}")

    return cfg

# ── Split CSV helpers ──────────────────────────────────────────────────────────
def _copy_split_csv(split_csv_path: str, run_dir: str):
    """Copy a single split CSV into the run directory."""
    if split_csv_path and os.path.isfile(split_csv_path):
        dest = os.path.join(run_dir, os.path.basename(split_csv_path))
        shutil.copy2(split_csv_path, dest)
        print(f"Split CSV saved : {dest}")
    elif split_csv_path:
        print(f"WARNING: split_csv not found, skipping copy: {split_csv_path}")

def _copy_split_csv_dir(split_csv_dir: str, run_dir: str):
    """
    Copy all CSV files from a split_csv_dir folder into the run directory,
    preserving the folder name so the record is self-contained.
    e.g. splits/split_train_val_test/ -> run_dir/split_train_val_test/
    """
    if not split_csv_dir or not os.path.isdir(split_csv_dir):
        if split_csv_dir:
            print(f"WARNING: split_csv_dir not found, "
                  f"skipping copy: {split_csv_dir}")
        return

    dest_dir = os.path.join(
        run_dir, os.path.basename(split_csv_dir.rstrip("/\\")))
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    for fname in os.listdir(split_csv_dir):
        if fname.endswith(".csv"):
            shutil.copy2(os.path.join(split_csv_dir, fname),
                         os.path.join(dest_dir, fname))
            copied += 1
    print(f"Split CSVs saved: {dest_dir}/  ({copied} file(s))")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.set:
        print("\nApplying overrides:")
        cfg = _apply_overrides(cfg, args.set)
    print_config(cfg)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device : {device}")
    eval_mode = cfg["training"].get("eval_mode", "train_val_test").lower()
    fold_mode = cfg["training"].get("fold_mode", "single")
    raw_seed   = cfg["training"].get("seed", 42)
    seed_list  = [int(s.strip()) for s in str(raw_seed).split(",")]
    base_seed  = seed_list[0]   # used for all non-training_only modes

    # ── Datetime-stamped run directory ────────────────────────────────────────
    run_dir = os.path.join(cfg["logging"]["log_dir"], datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # ── Save config snapshot ──────────────────────────────────────────────────
    saved_cfg_path = os.path.join(run_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        if args.set:
            f.write("# ── CLI overrides were applied to this config ──\n")
            for item in args.set:
                f.write(f"# --set {item}\n")
            f.write("\n")
        yaml.dump(dict(cfg), f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to: {saved_cfg_path}")

    # ── Save split CSV record ─────────────────────────────────────────────────
    split_csv_dir = cfg["dataset"].get("split_csv_dir", None)
    split_csv     = cfg["dataset"].get("split_csv",     None)

    if split_csv_dir:
        _copy_split_csv_dir(split_csv_dir, run_dir)
    elif split_csv:
        _copy_split_csv(split_csv, run_dir)
    else:
        print("Split source   : random (no split CSV)")

    # ── Dataset splits ────────────────────────────────────────────────────────
    splits = get_splits(cfg)
    print(f"Mode   : {fold_mode} | Folds: {len(splits)} | eval_mode: {eval_mode}")

    if eval_mode == "training_only":
        fold_seeds_preview = ", ".join(str(s) for s in seed_list[:len(splits)])
        print(f"Seed strategy  : per-fold  [{fold_seeds_preview}]  "
              f"(one seed per fold from config)")
    else:
        print(f"Seed strategy  : fixed  (seed={base_seed} for all folds)")

    fold_results = []
    for split in splits:
        fold_num     = split["fold"]
        fold_dir     = os.path.join(run_dir, f"fold_{fold_num}")
        ckpt_dir     = os.path.join(fold_dir, "checkpoints")

        train_loader = split["train_loader"]
        val_loader   = split.get("val_loader")
        test_loader  = split.get("test_loader")
        fold_csv     = split.get("split_csv")

        n_train = len(train_loader.dataset) if train_loader else 0
        n_val   = len(val_loader.dataset)   if val_loader   else 0
        n_test  = len(test_loader.dataset)  if test_loader  else 0

        # ── Per-fold seed ─────────────────────────────────────────────────
        seed = _fold_seed(base_seed, fold_num, eval_mode, seed_list)
        _set_seed(seed)

        print(f"\n{'='*55}")
        print(f"Fold {fold_num}/{len(splits)} | "
              f"train={n_train} val={n_val} test={n_test} | seed={seed}")
        if fold_csv:
            print(f"  Split CSV: {os.path.basename(fold_csv)}")
        print(f"{'='*55}")

        # Fresh model per fold — weight init uses seed set above
        model   = get_model(cfg)
        process = ModelProcess(model, cfg, device, checkpoint_dir=ckpt_dir, log_dir=fold_dir)
        process.logger.info(
            f"Fold {fold_num} | eval_mode={eval_mode} | seed={seed}"
            + (" [per-fold seed: training_only mode]"
               if eval_mode == "training_only" else ""))

        if args.resume and fold_num == 1:
            process.load_checkpoint(args.resume)

        process.train(train_loader, val_loader=val_loader, test_loader=test_loader)

        # ── Collect best metrics for summary ──────────────────────────────
        best_ckpt = torch.load(os.path.join(ckpt_dir, "best_model.pth"), map_location="cpu", weights_only=False)
        all_metrics = best_ckpt.get("metrics", {})

        # For train_val_test the final test row is appended to metrics.csv
        # post-loop (not stored in the checkpoint). Re-read it here.
        if eval_mode == "train_val_test":
            metrics_csv_path = os.path.join(fold_dir, "metrics.csv")
            if os.path.exists(metrics_csv_path):
                with open(metrics_csv_path, newline="") as _f:
                    rows = list(csv.DictReader(_f))
                final_row = next(
                    (r for r in reversed(rows)
                     if r.get("epoch", "").startswith("final")), {})
                test_cols = {k: float(v) for k, v in final_row.items()
                             if k.startswith("test_") and v}
                all_metrics.update(test_cols)

        # Gate summary columns by eval_mode
        def _keep(key: str) -> bool:
            if key.startswith("train_"): return True
            if key.startswith("val_"):
                return eval_mode in ("train_val_test", "train_val")
            if key.startswith("test_"):
                return eval_mode in ("train_val_test", "train_val", "train_test")
            return False

        fold_results.append(
            {"fold": fold_num, "seed": seed,
             **{k: v for k, v in all_metrics.items() if _keep(k)}})

    # ── Cross-fold summary ────────────────────────────────────────────────────
    if len(fold_results) > 1:
        summary_path = os.path.join(run_dir, "summary.csv")
        _write_summary(fold_results, summary_path)
        print(f"\nSummary written to {summary_path}")
    print("\nAll folds complete.")

def _write_summary(fold_results: list, path: str):
    """Write per-fold metrics + mean/std rows to summary CSV."""
    # seed column is informational only — exclude from mean/std calculation
    metric_keys = [k for k in fold_results[0].keys()
                   if k not in ("fold", "seed")]
    fieldnames  = ["fold", "seed"] + metric_keys

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in fold_results:
            writer.writerow(
                {k: (f"{v:.4f}" if isinstance(v, float) else v)
                 for k, v in row.items()})

        mean_row = {"fold": "mean", "seed": "-"}
        std_row  = {"fold": "std",  "seed": "-"}
        for k in metric_keys:
            vals = [r[k] for r in fold_results if isinstance(r.get(k), float)]
            if vals:
                mean_row[k] = f"{float(np.mean(vals)):.4f}"
                std_row[k]  = f"{float(np.std(vals)):.4f}"
        writer.writerow(mean_row)
        writer.writerow(std_row)

if __name__ == "__main__":
    main()
