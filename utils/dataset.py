"""
utils/dataset.py
----------------
Supports four dataset split modes driven entirely by config:

  fold_mode: single  + eval_mode: train_val_test  ->  train / val / test
  fold_mode: single  + eval_mode: train_val        ->  train / val  (val=test)
  fold_mode: single  + eval_mode: train_test        ->  train / test
  fold_mode: kfold   + eval_mode: train_val_test    ->  K folds; fixed holdout test
  fold_mode: kfold   + eval_mode: train_val         ->  K folds; held-out = val
  fold_mode: kfold   + eval_mode: train_test         ->  K folds; held-out = test
  fold_mode: *       + eval_mode: training_only      ->  train only

CSV-based splits (optional — two config keys accepted):
  dataset.split_csv_dir : path to a mode folder produced by split_generator.py
                          e.g.  splits/split_train_val_test/
                          Contains one *_fold{N}.csv per fold; loaded in order.
  dataset.split_csv     : path to a single CSV file (single-fold only).

  When either key is present the dataset is split using filenames from the
  CSV(s) instead of random splitting.  split_csv_dir takes priority.
  Omit both to use the existing random-split logic with no changes.

Public API
----------
  get_splits(cfg)  ->  list of dicts, one per fold:
      {
          "fold":         int,
          "train_loader": DataLoader,
          "val_loader":   DataLoader | None,
          "test_loader":  DataLoader | None,
          "split_csv":    str | None,   # path of CSV used, or None
      }
"""

import csv
import os
import re

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms.functional as TF

# ── Dataset ────────────────────────────────────────────────────────────────────
class SegmentationDataset(Dataset):
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    def __init__(self, images_dir, masks_dir, img_size=(512, 512), transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.img_size   = img_size
        self.transform  = transform

        self.image_paths = sorted([
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTENSIONS
        ])
        if not self.image_paths:
            raise RuntimeError(f"No images found in {images_dir}")

        self.mask_paths = []
        for img_path in self.image_paths:
            mask = self._find_mask(img_path.stem)
            if mask is None:
                raise FileNotFoundError(
                    f"No matching mask for '{img_path.name}' in {masks_dir}")
            self.mask_paths.append(mask)

    def _find_mask(self, stem):
        for ext in self.IMG_EXTENSIONS | {".png"}:
            c = self.masks_dir / f"{stem}{ext}"
            if c.exists():
                return c
        return None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask  = Image.open(self.mask_paths[idx]).convert("L")

        image = TF.resize(image, self.img_size, interpolation=Image.BILINEAR)
        mask  = TF.resize(mask,  self.img_size, interpolation=Image.NEAREST)

        if self.transform:
            image, mask = self.transform(image, mask)

        image   = TF.to_tensor(image)
        image   = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask_np = (np.array(mask, dtype=np.float32) / 255).round().astype(np.int64)
        return image, torch.from_numpy(mask_np)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _make_loader(dataset, indices, batch_size, num_workers, shuffle):
    if not indices:
        return None
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

def _loader_kw(cfg):
    return dict(batch_size=cfg["training"]["batch_size"], num_workers=cfg["dataset"].get("num_workers", 4))

def _split_indices(n, fractions, seed=42):
    """Split range(n) into len(fractions) index lists."""
    rng    = np.random.default_rng(seed)
    idx    = rng.permutation(n).tolist()
    splits, cursor = [], 0
    for i, frac in enumerate(fractions):
        size = int(n * frac) if i < len(fractions) - 1 else n - cursor
        splits.append(idx[cursor: cursor + size])
        cursor += size
    return splits

def _build_datasets(cfg):
    from utils.augmentations import get_train_augmentations, get_val_augmentations
    ds_cfg   = cfg["dataset"]
    img_size = tuple(ds_cfg.get("img_size", [512, 512]))
    kw       = dict(images_dir=ds_cfg["images_dir"],
                    masks_dir=ds_cfg["masks_dir"],
                    img_size=img_size)
    train_ds = SegmentationDataset(
        **kw,
        transform=get_train_augmentations(img_size)
                  if ds_cfg.get("augment", False) else get_val_augmentations())
    base_ds  = SegmentationDataset(**kw, transform=get_val_augmentations())
    return train_ds, base_ds

# ── CSV helpers ────────────────────────────────────────────────────────────────
def _load_csv_indices(csv_path: str, dataset: SegmentationDataset, col: str) -> List[int]:
    """
    Read filenames from a CSV column and return the corresponding dataset indices.
    Matches by basename; empty/padding rows are silently skipped.
    Raises FileNotFoundError if any non-empty filename is absent from the dataset.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return []

    if col not in rows[0]:
        raise ValueError(
            f"Column '{col}' not found in {csv_path}. "
            f"Available columns: {list(rows[0].keys())}")

    name_to_idx = {p.name: i for i, p in enumerate(dataset.image_paths)}
    indices, missing = [], []
    for row in rows:
        fname = row[col].strip()
        if not fname:
            continue
        if fname not in name_to_idx:
            missing.append(fname)
        else:
            indices.append(name_to_idx[fname])

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} filename(s) from CSV column '{col}' not found "
            f"in images_dir.  First few: {missing[:5]}")
    return indices

def _sorted_csv_files(folder: str) -> List[str]:
    """
    Return absolute paths of all CSV files in folder, sorted by fold number
    so fold1 < fold2 < ... < fold10 (numeric, not lexicographic).
    """
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.startswith("split_") and f.endswith(".csv")
        #if f.endswith(".csv")
    ]
    def _fold_num(p):
        m = re.search(r"fold(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0

    return sorted(files, key=_fold_num)

def _csv_headers(csv_path: str) -> List[str]:
    with open(csv_path, newline="") as f:
        return next(csv.reader(f))

def _validate_csv_columns(csv_path: str, eval_mode: str):
    """Raise ValueError if the CSV is missing columns required for eval_mode."""
    required = {
        "train_val_test": ["train", "val", "test"],
        "train_val":      ["train", "val"],
        "train_test":     ["train", "test"],
        "training_only":  ["train"],
    }[eval_mode]
    headers      = _csv_headers(csv_path)
    missing_cols = [c for c in required if c not in headers]
    if missing_cols:
        raise ValueError(
            f"CSV '{os.path.basename(csv_path)}' is missing columns "
            f"{missing_cols} required for eval_mode='{eval_mode}'.")
    return headers

def _build_fold_from_csv(cfg, eval_mode: str, csv_path: str, fold_num: int, train_ds: SegmentationDataset, base_ds:  SegmentationDataset, kw: dict) -> dict:
    """
    Build one fold dict from a single CSV file.
    The mask for each image is already paired inside SegmentationDataset.__init__()
    by stem matching; the CSV only controls which indices go into each split.
    """
    headers = _validate_csv_columns(csv_path, eval_mode)

    tr_i  = _load_csv_indices(csv_path, base_ds, "train")
    val_i = _load_csv_indices(csv_path, base_ds, "val")  if "val"  in headers else []
    te_i  = _load_csv_indices(csv_path, base_ds, "test") if "test" in headers else []

    print(f"  Fold {fold_num}: {os.path.basename(csv_path)}"
          f"  [train={len(tr_i)} val={len(val_i)} test={len(te_i)}]")

    if eval_mode == "training_only":
        return {"fold": fold_num,
                "train_loader": _make_loader(train_ds, tr_i, shuffle=True,  **kw),
                "val_loader":   None,
                "test_loader":  None,
                "split_csv":    csv_path}

    elif eval_mode == "train_val":
        val_loader = _make_loader(base_ds, val_i, shuffle=False, **kw)
        return {"fold": fold_num,
                "train_loader": _make_loader(train_ds, tr_i, shuffle=True, **kw),
                "val_loader":   val_loader,
                "test_loader":  val_loader,   # val doubles as test
                "split_csv":    csv_path}

    elif eval_mode == "train_val_test":
        return {"fold": fold_num,
                "train_loader": _make_loader(train_ds, tr_i,  shuffle=True,  **kw),
                "val_loader":   _make_loader(base_ds,  val_i, shuffle=False, **kw),
                "test_loader":  _make_loader(base_ds,  te_i,  shuffle=False, **kw),
                "split_csv":    csv_path}

    else:  # train_test
        return {"fold": fold_num,
                "train_loader": _make_loader(train_ds, tr_i, shuffle=True,  **kw),
                "val_loader":   None,
                "test_loader":  _make_loader(base_ds,  te_i, shuffle=False, **kw),
                "split_csv":    csv_path}

# ── Public API ─────────────────────────────────────────────────────────────────
def get_splits(cfg) -> List[dict]:
    """
    Returns list of fold dicts:
      [{"fold": 1, "train_loader": ..., "val_loader": ...,
        "test_loader": ..., "split_csv": str|None}, ...]

    Priority order for split source:
      1. dataset.split_csv_dir  — folder of per-fold CSVs (multi-fold CSV splits)
      2. dataset.split_csv      — single CSV file         (single-fold CSV split)
      3. random splitting       — existing behaviour, no CSV needed
    """
    fold_mode     = cfg["training"].get("fold_mode", "single").lower()
    eval_mode     = cfg["training"].get("eval_mode", "train_val_test").lower()
    split_csv_dir = cfg["dataset"].get("split_csv_dir", None)
    split_csv     = cfg["dataset"].get("split_csv",     None)

    valid_modes = {"train_val_test", "train_val", "train_test", "training_only"}
    if eval_mode not in valid_modes:
        raise ValueError(f"Unknown eval_mode: '{eval_mode}'. "
                         f"Options: {sorted(valid_modes)}")

    # ── 1. CSV folder: one CSV per fold ──────────────────────────────────────
    if split_csv_dir:
        return _get_splits_from_csv_dir(cfg, eval_mode, split_csv_dir)

    # ── 2. Single CSV file ────────────────────────────────────────────────────
    if split_csv:
        return _get_splits_from_csv_file(cfg, eval_mode, split_csv)

    # ── 3. Random splitting (original behaviour, unchanged) ───────────────────
    if fold_mode == "single":
        return _single_fold(cfg, eval_mode)
    elif fold_mode == "kfold":
        return _kfold(cfg, eval_mode)
    else:
        raise ValueError(f"Unknown fold_mode: '{fold_mode}'. Use 'single' or 'kfold'.")

def _get_splits_from_csv_dir(cfg, eval_mode: str, split_csv_dir: str) -> List[dict]:
    """
    Load one CSV per fold from a directory (e.g. splits/split_train_val_test/).
    CSVs are matched in ascending fold-number order via _sorted_csv_files().
    """
    if not os.path.isdir(split_csv_dir):
        raise FileNotFoundError(f"split_csv_dir not found: {split_csv_dir}")

    csv_files = _sorted_csv_files(split_csv_dir)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in: {split_csv_dir}")

    print(f"Using CSV split folder : {split_csv_dir}")
    print(f"  Found {len(csv_files)} CSV file(s)  (eval_mode={eval_mode})")

    train_ds, base_ds = _build_datasets(cfg)
    kw = _loader_kw(cfg)

    return [_build_fold_from_csv(cfg, eval_mode, csv_path, fold_num, train_ds, base_ds, kw) for fold_num, csv_path in enumerate(csv_files, 1)]

def _get_splits_from_csv_file(cfg, eval_mode: str, split_csv: str) -> List[dict]:
    """Load a single CSV file as a one-fold split."""
    if not os.path.isfile(split_csv):
        raise FileNotFoundError(f"split_csv not found: {split_csv}")

    print(f"Using CSV split : {split_csv}  (eval_mode={eval_mode})")
    train_ds, base_ds = _build_datasets(cfg)
    kw = _loader_kw(cfg)
    return [_build_fold_from_csv(cfg, eval_mode, split_csv, 1, train_ds, base_ds, kw)]

# ── Random split implementations (original, unchanged) ────────────────────────
def _single_fold(cfg, eval_mode):
    train_ds, base_ds = _build_datasets(cfg)
    kw = _loader_kw(cfg)
    n  = len(base_ds)
    ds = cfg["dataset"]

    no_csv = {"split_csv": None}

    if eval_mode == "training_only":
        return [{"fold": 1,
                 "train_loader": _make_loader(train_ds, list(range(n)), shuffle=True, **kw),
                 "val_loader":  None,
                 "test_loader": None,
                 **no_csv}]

    elif eval_mode == "train_val":
        val       = ds.get("val_split", 0.2)
        tr_i, val_i = _split_indices(n, [1 - val, val])
        val_loader  = _make_loader(base_ds, val_i, shuffle=False, **kw)
        return [{"fold": 1,
                 "train_loader": _make_loader(train_ds, tr_i, shuffle=True, **kw),
                 "val_loader":   val_loader,
                 "test_loader":  val_loader,
                 **no_csv}]

    elif eval_mode == "train_val_test":
        te  = ds.get("test_split", 0.2)
        val = ds.get("val_split",  0.1)
        tr_i, val_i, te_i = _split_indices(n, [1 - te - val, val, te])
        return [{"fold": 1,
                 "train_loader": _make_loader(train_ds, tr_i,  shuffle=True,  **kw),
                 "val_loader":   _make_loader(base_ds,  val_i, shuffle=False, **kw),
                 "test_loader":  _make_loader(base_ds,  te_i,  shuffle=False, **kw),
                 **no_csv}]

    else:  # train_test
        te = ds.get("test_split", 0.2)
        tr_i, te_i = _split_indices(n, [1 - te, te])
        return [{"fold": 1,
                 "train_loader": _make_loader(train_ds, tr_i, shuffle=True,  **kw),
                 "val_loader":   None,
                 "test_loader":  _make_loader(base_ds,  te_i, shuffle=False, **kw),
                 **no_csv}]

def _kfold(cfg, eval_mode):
    from sklearn.model_selection import KFold
    train_ds, base_ds = _build_datasets(cfg)
    kw      = _loader_kw(cfg)
    n       = len(base_ds)
    n_folds = cfg["training"].get("n_folds", 5)
    ds      = cfg["dataset"]
    kf      = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    no_csv  = {"split_csv": None}

    if eval_mode == "training_only":
        tr_i = list(range(n))
        return [{"fold": f + 1,
                 "train_loader": _make_loader(train_ds, tr_i, shuffle=True, **kw),
                 "val_loader":   None,
                 "test_loader":  None,
                 **no_csv}
                for f in range(n_folds)]

    elif eval_mode == "train_val":
        all_idx = list(range(n))
        return [{"fold": fold_num,
                 "train_loader": _make_loader(train_ds, list(tr),  shuffle=True,  **kw),
                 "val_loader":   _make_loader(base_ds,  list(val), shuffle=False, **kw),
                 "test_loader":  _make_loader(base_ds,  list(val), shuffle=False, **kw),
                 **no_csv}
                for fold_num, (tr, val) in enumerate(kf.split(all_idx), 1)]

    elif eval_mode == "train_val_test":
        te         = ds.get("test_split", 0.2)
        tv_i, te_i = _split_indices(n, [1 - te, te])
        tv_arr     = np.array(tv_i)
        return [{"fold": fold_num,
                 "train_loader": _make_loader(train_ds, tv_arr[tr].tolist(),  shuffle=True,  **kw),
                 "val_loader":   _make_loader(base_ds,  tv_arr[val].tolist(), shuffle=False, **kw),
                 "test_loader":  _make_loader(base_ds,  te_i,                 shuffle=False, **kw),
                 **no_csv}
                for fold_num, (tr, val) in enumerate(kf.split(tv_arr), 1)]

    else:  # train_test
        all_idx = list(range(n))
        return [{"fold": fold_num,
                 "train_loader": _make_loader(train_ds, list(tr), shuffle=True,  **kw),
                 "val_loader":   None,
                 "test_loader":  _make_loader(base_ds,  list(te), shuffle=False, **kw),
                 **no_csv}
                for fold_num, (tr, te) in enumerate(kf.split(all_idx), 1)]

# Legacy shim so existing imports don't break
def build_dataloaders(cfg):
    splits = get_splits(cfg)
    s = splits[0]
    return s["train_loader"], s.get("val_loader"), s.get("test_loader")