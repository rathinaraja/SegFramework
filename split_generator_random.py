"""
utils/split_generator.py
-------------------------
Scans an image folder (recursively), collects all image filenames, and writes
per-fold CSV split files for each eval_mode so every model trains on identical
samples across all folds.

Output structure
----------------
splits/
  split_train_val_test/
      split_train_val_test_{dataset_name}_fold1.csv
      split_train_val_test_{dataset_name}_fold2.csv
      ...
  split_train_val/
      split_train_val_{dataset_name}_fold1.csv
      ...
  split_train_test/
      split_train_test_{dataset_name}_fold1.csv
      ...
  split_training_only/
      split_training_only_{dataset_name}_fold1.csv
      ...

CSV format — one column per split, rows are filenames, shorter columns
padded with "" so all columns are the same length:
  train_val_test  ->  columns: train, val, test
  train_val       ->  columns: train, val
  train_test      ->  columns: train, test
  training_only   ->  columns: train

Splitting strategy
------------------
  folds=1  : single random split using val_split / test_split fractions.
  folds>1  : KFold strategy per mode:
             train_val_test : carve out test_split as a fixed holdout first,
                              then KFold the remainder into train / val folds.
             train_val      : KFold; held-out fold = val  (no test column).
             train_test     : KFold; held-out fold = test (no val column).
             training_only  : KFold; all data used as train every fold.

Usage (CLI)
-----------
    python utils/split_generator.py \\
        --images_dir /data/patches \\
        --output_dir splits/ \\
        --dataset_name my_dataset \\
        --folds      5 \\
        --val_split  0.1 \\
        --test_split 0.2 \\
        --seed       42

    # Single fold, specific modes only
    python utils/split_generator.py \\
        --images_dir /data/patches \\
        --output_dir splits/ \\
        --dataset_name my_dataset \\
        --folds 1 \\
        --modes train_val_test train_val

Usage (API)
-----------
    from utils.split_generator import generate_splits
    written = generate_splits(
        images_dir   = "/data/patches",
        output_dir   = "splits/",
        dataset_name = "my_dataset",
        n_folds      = 5,
        val_split    = 0.1,
        test_split   = 0.2,
        seed         = 42,
        modes        = None,   # None -> all four modes
    )
    # written: dict[mode -> list[csv_path]]  (one path per fold)
"""

import argparse
import csv
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
ALL_MODES = ["train_val_test", "train_val", "train_test", "training_only"]
MODE_COLUMNS = {
    "train_val_test": ["train", "val", "test"],
    "train_val":      ["train", "val"],
    "train_test":     ["train", "test"],
    "training_only":  ["train"],
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def _collect_images(images_dir: str) -> List[str]:
    """Recursively collect all image filenames (basename only), sorted."""
    root = Path(images_dir)
    if not root.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    files = sorted([
        p.name
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    ])
    if not files:
        raise RuntimeError(f"No image files found under: {images_dir}")
    return files

def _shuffle(items: List[str], seed: int) -> List[str]:
    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out

def _fraction_split(items: List[str], val_frac: float, test_frac: float) -> Tuple[List[str], List[str], List[str]]:
    """Split a list into (train, val, test) by fractions."""
    n       = len(items)
    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)
    n_train = n - n_test - n_val
    return (items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :])

def _kfold_indices(n: int, n_folds: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    """Return list of (train_indices, held_out_indices) for each fold."""
    try:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return [(list(tr), list(ho)) for tr, ho in kf.split(range(n))]
    except ImportError:
        # Manual KFold fallback (no sklearn required)
        rng     = random.Random(seed)
        idx     = list(range(n))
        rng.shuffle(idx)
        fold_sz = n // n_folds
        folds   = [idx[i * fold_sz : (i + 1) * fold_sz] for i in range(n_folds)]
        folds[-1] += idx[n_folds * fold_sz:]   # last fold absorbs remainder
        result = []
        for i in range(n_folds):
            ho = folds[i]
            tr = [x for j, f in enumerate(folds) if j != i for x in f]
            result.append((tr, ho))
        return result

def _write_csv(path: str, columns: Dict[str, List[str]]):
    """Write CSV with equal-length columns, padding shorter ones with ''."""
    max_len = max((len(v) for v in columns.values()), default=0)
    headers = list(columns.keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for i in range(max_len):
            writer.writerow({
                h: (columns[h][i] if i < len(columns[h]) else "")
                for h in headers
            })
    counts = " | ".join(f"{h}={len(columns[h])}" for h in headers)
    print(f"    {os.path.basename(path)}  [{counts}]")

def _build_fold_splits(
    all_files  : List[str],
    mode       : str,
    n_folds    : int,
    val_split  : float,
    test_split : float,
    seed       : int,
) -> List[Dict[str, List[str]]]:
    """
    Return a list of n_folds dicts, each mapping split-name -> list of filenames.
    """
    result   = []
    shuffled = _shuffle(all_files, seed)

    if n_folds == 1:
        # ── Single fold: simple fraction split ────────────────────────────
        train, val, test = _fraction_split(shuffled, val_split, test_split)
        data: Dict[str, List[str]] = {}
        if "train" in MODE_COLUMNS[mode]: data["train"] = train
        if "val"   in MODE_COLUMNS[mode]: data["val"]   = val
        if "test"  in MODE_COLUMNS[mode]: data["test"]  = test
        result.append(data)

    else:
        # ── Multi-fold: KFold strategy per mode ───────────────────────────
        if mode == "train_val_test":
            # Fixed holdout test set, KFold on the rest for train / val
            _, _, test = _fraction_split(shuffled, 0.0, test_split)
            tv         = shuffled[: len(shuffled) - len(test)]
            for tr_idx, val_idx in _kfold_indices(len(tv), n_folds, seed):
                result.append({
                    "train": [tv[i] for i in tr_idx],
                    "val":   [tv[i] for i in val_idx],
                    "test":  test,           # same holdout every fold
                })

        elif mode == "train_val":
            # KFold: held-out fold = val
            for tr_idx, ho_idx in _kfold_indices(len(shuffled), n_folds, seed):
                result.append({
                    "train": [shuffled[i] for i in tr_idx],
                    "val":   [shuffled[i] for i in ho_idx],
                })

        elif mode == "train_test":
            # KFold: held-out fold = test
            for tr_idx, ho_idx in _kfold_indices(len(shuffled), n_folds, seed):
                result.append({
                    "train": [shuffled[i] for i in tr_idx],
                    "test":  [shuffled[i] for i in ho_idx],
                })

        else:  # training_only
            # KFold: all data = train each fold (held-out not used)
            for tr_idx, _ in _kfold_indices(len(shuffled), n_folds, seed):
                result.append({
                    "train": [shuffled[i] for i in tr_idx],
                })

    return result

# ── Public API ─────────────────────────────────────────────────────────────────
def generate_splits(
    images_dir   : str,
    output_dir   : str,
    dataset_name : str,
    n_folds      : int             = 1,
    val_split    : float           = 0.1,
    test_split   : float           = 0.2,
    seed         : int             = 42,
    modes        : Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Generate per-fold split CSV files for the requested modes.
    Parameters
    ----------
    images_dir   : root folder to scan for images (recursive)
    output_dir   : root folder where mode sub-folders are written
    dataset_name : label used in CSV filenames
    n_folds      : number of folds (1 = single fraction split)
    val_split    : validation fraction (single-fold / train_val_test KFold)
    test_split   : test fraction (single-fold / train_val_test KFold holdout)
    seed         : random seed for reproducibility
    modes        : list of modes to generate; None -> all four
    Returns
    -------
    dict mapping mode_name -> list of absolute CSV paths (one per fold)
    """
    modes = modes or ALL_MODES
    invalid = set(modes) - set(ALL_MODES)
    if invalid:
        raise ValueError(f"Unknown modes: {invalid}.  Valid: {ALL_MODES}")
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")

    all_files = _collect_images(images_dir)

    print(f"\nImages found : {len(all_files)}")
    print(f"Dataset name : {dataset_name}")
    print(f"Folds        : {n_folds}")
    print(f"Val split    : {val_split}")
    print(f"Test split   : {test_split}")
    print(f"Seed         : {seed}")
    print(f"Output dir   : {os.path.abspath(output_dir)}\n")

    written: Dict[str, List[str]] = {}

    for mode in modes:
        mode_dir    = os.path.join(output_dir, f"split_{mode}")
        fold_splits = _build_fold_splits(
            all_files, mode, n_folds, val_split, test_split, seed)

        print(f"  [{mode}]  ->  {mode_dir}/")
        paths = []
        for fold_num, data in enumerate(fold_splits, 1):
            fname = f"split_{mode}_{dataset_name}_fold{fold_num}.csv"
            path  = os.path.join(mode_dir, fname)
            _write_csv(path, data)
            paths.append(os.path.abspath(path))

        written[mode] = paths

    total = sum(len(v) for v in written.values())
    print(f"\nDone. {total} CSV file(s) written across {len(written)} mode(s).")
    return written

# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Generate deterministic per-fold train/val/test split CSVs.")
    p.add_argument("--images_dir",   type=str, required=True, help="Root folder to scan for images (searched recursively)")
    p.add_argument("--output_dir",   type=str, default="splits", help="Root folder for CSV sub-folders (default: splits/)")
    p.add_argument("--dataset_name", type=str, required=True, help="Dataset label used in CSV filenames")
    p.add_argument("--folds",        type=int, default=1, help="Number of folds (default: 1)")
    p.add_argument("--val_split",    type=float, default=0.1, help="Val fraction for single-fold / KFold train_val_test (default: 0.1)")
    p.add_argument("--test_split",   type=float, default=0.2, help="Test fraction (default: 0.2)")
    p.add_argument("--seed",         type=int,   default=42, help="Random seed (default: 42)")
    p.add_argument("--modes",        nargs="*",  default=None, metavar="MODE", help=f"Modes to generate (default: all). " f"Options: {' | '.join(ALL_MODES)}")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    generate_splits(
        images_dir   = args.images_dir,
        output_dir   = args.output_dir,
        dataset_name = args.dataset_name,
        n_folds      = args.folds,
        val_split    = args.val_split,
        test_split   = args.test_split,
        seed         = args.seed,
        modes        = args.modes,
    )
