"""
utils/split_generator.py
-------------------------
Scans an image folder (recursively), groups images by their unique slide/patient
name, and writes per-fold CSV split files for each eval_mode with ZERO DATA
LEAKAGE — all tiles from the same slide always go to the same split.

Image filename pattern
----------------------
    {slide_name}_x{XXXXXX}_y{YYYYYY}.ext
Examples:
    S08-38628 G11_x067376_y034486.png      -> slide: S08-38628 G11
    SP-22-078912 A11-1_x012345_y067890.png -> slide: SP-22-078912 A11-1

Splitting is done at the SLIDE level:
  1. Collect all unique slide names
  2. Shuffle slides with a fixed seed
  3. Split SLIDES into train/val/test groups by fraction or KFold
  4. Expand each slide group to its full list of tile filenames
This guarantees a tile from slide X never appears in two different splits.

Output structure
----------------
splits/
  dataset_summary_{dataset_name}.csv          <- slide inventory
  split_train_val_test/
      split_train_val_test_{dataset_name}_fold1.csv   (cols: train, val, test)
      slide_split_train_val_test_{dataset_name}_fold1.csv  (slide assignment)
      ...
  split_train_val/
      split_train_val_{dataset_name}_fold1.csv        (cols: train, val)
      slide_split_train_val_{dataset_name}_fold1.csv
      ...
  split_train_test/ ...
  split_training_only/ ...

Usage (CLI)
-----------
    python utils/split_generator.py \\
        --images_dir /data/patches \\
        --output_dir splits/ \\
        --dataset_name SRC \\
        --folds      5 \\
        --val_split  0.1 \\
        --test_split 0.2 \\
        --seed       42
"""

import argparse
import csv
import os
import random
import re
from collections import defaultdict
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
# Extract slide name: everything before the LAST _x{digits}_y{digits} block
# Handles spaces and special chars in slide names (e.g. "S08-38628 G11")
# _TILE_RE = re.compile(r"^(.+)_x\d+_y\d+$", re.IGNORECASE)
_TILE_RE = re.compile(r"^(.+)_x\d+_y\d+(?:_augmented)?$", re.IGNORECASE)

# ── Collection ─────────────────────────────────────────────────────────────────
def _collect_and_group(images_dir: str,) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Recursively collect all image files and group by slide name.
    The slide name is extracted from the filename stem by stripping the
    trailing _x{digits}_y{digits} coordinate suffix.
    Returns
    -------
    groups    : dict  slide_name -> sorted list of filenames (basenames only)
    ungrouped : list  of filenames that don't match the tile pattern
                (appended to train in every fold; never leaked)
    """
    root = Path(images_dir)
    if not root.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    all_files = sorted([
        p.name
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    ])
    if not all_files:
        raise RuntimeError(f"No image files found under: {images_dir}")

    groups: Dict[str, List[str]] = defaultdict(list)
    ungrouped: List[str] = []

    for fname in all_files:
        stem  = Path(fname).stem          # filename without extension
        match = _TILE_RE.match(stem)
        if match:
            groups[match.group(1)].append(fname)
        else:
            ungrouped.append(fname)
    return {k: sorted(v) for k, v in groups.items()}, ungrouped

# ── Splitting at SLIDE level ───────────────────────────────────────────────────
def _shuffle(items: list, seed: int) -> list:
    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out

def _fraction_split(slides: List[str], val_frac: float, test_frac: float) -> Tuple[List[str], List[str], List[str]]:
    """
    Split a list of SLIDE NAMES into (train, val, test) by fraction.
    Fractions apply to the number of slides, not images.
    """
    n      = len(slides)
    n_test = max(1, round(n * test_frac)) if test_frac > 0 else 0
    n_val  = max(1, round(n * val_frac))  if val_frac  > 0 else 0
    # Clamp so we always have at least 1 slide in train
    n_val  = min(n_val,  n - n_test - 1)
    n_test = min(n_test, n - n_val  - 1)
    if n - n_val - n_test < 1:
        raise ValueError(
            f"Only {n} slides — too few for val={val_frac}, test={test_frac}. "
            f"Reduce fractions or use more slides.")
    train = slides[: n - n_val - n_test]
    val   = slides[n - n_val - n_test : n - n_test]
    test  = slides[n - n_test :]
    return train, val, test

def _kfold(slides: List[str], n_folds: int, seed: int) -> List[Tuple[List[str], List[str]]]:
    """
    KFold split of SLIDE NAMES.
    Returns list of (train_slides, held_out_slides) per fold.
    """
    try:
        from sklearn.model_selection import KFold
        kf  = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        idx = list(range(len(slides)))
        return [([slides[i] for i in tr], [slides[i] for i in ho])
                for tr, ho in kf.split(idx)]
    except ImportError:
        # Manual fallback — no sklearn required
        rng     = random.Random(seed)
        shuf    = list(slides)
        rng.shuffle(shuf)
        sz      = len(shuf) // n_folds
        folds   = [shuf[i*sz:(i+1)*sz] for i in range(n_folds)]
        folds[-1] += shuf[n_folds*sz:]
        return [([x for j,f in enumerate(folds) if j!=i for x in f], folds[i])
                for i in range(n_folds)]

def _expand(slide_names: List[str], groups: Dict[str, List[str]]) -> List[str]:
    """Expand slide names to their full sorted list of tile filenames."""
    out = []
    for name in slide_names:
        out.extend(groups.get(name, []))
    return out

# ── Build all fold splits ──────────────────────────────────────────────────────
def _build_folds(
    groups    : Dict[str, List[str]],
    ungrouped : List[str],
    mode      : str,
    n_folds   : int,
    val_split : float,
    test_split: float,
    seed      : int,
) -> Tuple[List[Dict[str, List[str]]], List[Dict[str, List[str]]]]:
    """
    Build per-fold split dicts at SLIDE level, then expand to tile filenames.
    Returns
    -------
    image_folds : list of dicts  split_name -> list[tile_filename]
    slide_folds : list of dicts  split_name -> list[slide_name]
                  (used for the slide-assignment CSV)
    """
    all_slides  = _shuffle(list(groups.keys()), seed)
    image_folds = []
    slide_folds = []

    if n_folds == 1:
        tr_s, val_s, te_s = _fraction_split(all_slides, val_split, test_split)

        s_data: Dict[str, List[str]] = {"train": tr_s}
        i_data: Dict[str, List[str]] = {"train": _expand(tr_s, groups) + ungrouped}
        if "val" in MODE_COLUMNS[mode]:
            s_data["val"] = val_s
            i_data["val"] = _expand(val_s, groups)
        if "test" in MODE_COLUMNS[mode]:
            s_data["test"] = te_s
            i_data["test"] = _expand(te_s, groups)

        image_folds.append(i_data)
        slide_folds.append(s_data)

    else:
        if mode == "train_val_test":
            # Carve out a fixed holdout test set of SLIDES first
            _, _, te_slides = _fraction_split(all_slides, 0.0, test_split)
            tv_slides       = all_slides[: len(all_slides) - len(te_slides)]
            te_files        = _expand(te_slides, groups)

            for tr_s, val_s in _kfold(tv_slides, n_folds, seed):
                image_folds.append({
                    "train": _expand(tr_s,  groups) + ungrouped,
                    "val":   _expand(val_s, groups),
                    "test":  te_files,
                })
                slide_folds.append({
                    "train": tr_s,
                    "val":   val_s,
                    "test":  te_slides,
                })

        elif mode == "train_val":
            for tr_s, ho_s in _kfold(all_slides, n_folds, seed):
                image_folds.append({
                    "train": _expand(tr_s, groups) + ungrouped,
                    "val":   _expand(ho_s, groups),
                })
                slide_folds.append({"train": tr_s, "val": ho_s})

        elif mode == "train_test":
            for tr_s, ho_s in _kfold(all_slides, n_folds, seed):
                image_folds.append({
                    "train": _expand(tr_s, groups) + ungrouped,
                    "test":  _expand(ho_s, groups),
                })
                slide_folds.append({"train": tr_s, "test": ho_s})

        else:  # training_only
            for tr_s, _ in _kfold(all_slides, n_folds, seed):
                image_folds.append({
                    "train": _expand(tr_s, groups) + ungrouped,
                })
                slide_folds.append({"train": tr_s})

    return image_folds, slide_folds

# ── Leakage verification ───────────────────────────────────────────────────────
def _verify_no_leakage(slide_folds : List[Dict[str, List[str]]], mode : str,):
    """
    Verify at SLIDE level that no slide appears in more than one split.
    Raises AssertionError immediately if leakage is found.
    """
    cols = MODE_COLUMNS[mode]
    leaked = False
    for fold_num, sf in enumerate(slide_folds, 1):
        seen: Dict[str, str] = {}
        for col in cols:
            for slide in sf.get(col, []):
                if not slide:
                    continue
                if slide in seen:
                    print(f"  !! LEAKAGE fold {fold_num}: '{slide}' "
                          f"in '{seen[slide]}' AND '{col}'")
                    leaked = True
                else:
                    seen[slide] = col
    if leaked:
        raise AssertionError(
            "Data leakage detected — slides appear in multiple splits. "
            "This should not happen; please file a bug report.")
    print(f"    Leakage check : PASSED  ({len(slide_folds)} fold(s))")

# ── CSV writers ────────────────────────────────────────────────────────────────
def _write_image_csv(path: str, data: Dict[str, List[str]]):
    """Write image-filename CSV — one column per split, padded to equal length."""
    max_len = max((len(v) for v in data.values()), default=0)
    headers = list(data.keys())
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for i in range(max_len):
            writer.writerow({
                h: (data[h][i] if i < len(data[h]) else "")
                for h in headers
            })
    counts = " | ".join(f"{h}={len(data[h])}" for h in headers)
    print(f"      images : {os.path.basename(path)}  [{counts}]")

def _write_slide_csv(path: str, data: Dict[str, List[str]]):
    """
    Write slide-assignment CSV.
    Columns: slide, split  (one row per slide — no padding needed)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slide", "split"])
        writer.writeheader()
        for split_name, slides in data.items():
            for slide in sorted(slides):
                writer.writerow({"slide": slide, "split": split_name})
    total = sum(len(v) for v in data.values())
    counts = " | ".join(f"{k}={len(v)}" for k, v in data.items())
    print(f"      slides : {os.path.basename(path)}  [{counts}]  total={total}")

def _write_dataset_summary(path      : str, groups    : Dict[str, List[str]], ungrouped : List[str],):
    """
    Write a dataset inventory CSV:
      slide, n_images
    One row per slide, sorted by slide name, plus a summary row for ungrouped files.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slide", "n_images"])
        writer.writeheader()
        for slide in sorted(groups.keys()):
            writer.writerow({"slide": slide, "n_images": len(groups[slide])})
        if ungrouped:
            writer.writerow({"slide": "__ungrouped__", "n_images": len(ungrouped)})
    total = sum(len(v) for v in groups.values()) + len(ungrouped)
    print(f"  Dataset summary: {path}")
    print(f"    {len(groups)} slides | {total} images total"
          + (f" | {len(ungrouped)} ungrouped" if ungrouped else ""))

# ── Public API ─────────────────────────────────────────────────────────────────
def generate_splits(
    images_dir   : str,
    output_dir   : str,
    dataset_name : str,
    n_folds      : int                   = 1,
    val_split    : float                 = 0.1,
    test_split   : float                 = 0.2,
    seed         : int                   = 42,
    modes        : Optional[List[str]]   = None,
) -> Dict[str, List[str]]:
    """
    Generate per-fold split CSV files with guaranteed zero data leakage.
    Splitting is done at the SLIDE level — all tiles from the same slide
    always land in the same split (train, val, or test).
    Parameters
    ----------
    images_dir   : root folder to scan for images (recursive)
    output_dir   : root folder where mode sub-folders are written
    dataset_name : label used in CSV filenames
    n_folds      : number of folds (1 = single fraction split)
    val_split    : fraction of SLIDES for validation
    test_split   : fraction of SLIDES for test
    seed         : random seed — same seed = identical splits every run
    modes        : list of modes to generate; None -> all four
    Returns
    -------
    dict  mode_name -> list of absolute image-CSV paths (one per fold)
    """
    modes = modes or ALL_MODES
    invalid = set(modes) - set(ALL_MODES)
    if invalid:
        raise ValueError(f"Unknown modes: {invalid}.  Valid: {ALL_MODES}")
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")

    groups, ungrouped = _collect_and_group(images_dir)
    n_slides = len(groups)
    n_images = sum(len(v) for v in groups.values()) + len(ungrouped)

    print(f"\n{'='*60}")
    print(f" Split Generator — slide-level splitting (zero leakage)")
    print(f"{'='*60}")
    print(f"  Unique slides  : {n_slides}")
    print(f"  Total images   : {n_images}")
    if ungrouped:
        print(f"  Ungrouped      : {len(ungrouped)} (appended to train, not split)")
    print(f"  Dataset name   : {dataset_name}")
    print(f"  Folds          : {n_folds}")
    print(f"  Val split      : {val_split}  (~{round(n_slides*val_split)} slides)")
    print(f"  Test split     : {test_split}  (~{round(n_slides*test_split)} slides)")
    print(f"  Seed           : {seed}")
    print(f"  Output dir     : {os.path.abspath(output_dir)}")
    print(f"{'='*60}\n")

    # ── Dataset summary CSV ───────────────────────────────────────────────────
    summary_path = os.path.join(
        output_dir, f"dataset_summary_{dataset_name}.csv")
    _write_dataset_summary(summary_path, groups, ungrouped)

    written: Dict[str, List[str]] = {}

    for mode in modes:
        mode_dir             = os.path.join(output_dir, f"split_{mode}")
        image_folds, slide_folds = _build_folds(
            groups, ungrouped, mode, n_folds, val_split, test_split, seed)

        print(f"\n  [{mode}]")
        _verify_no_leakage(slide_folds, mode)

        image_paths = []
        for fold_num, (i_data, s_data) in enumerate(
                zip(image_folds, slide_folds), 1):

            img_fname   = f"split_{mode}_{dataset_name}_fold{fold_num}.csv"
            slide_fname = f"slide_split_{mode}_{dataset_name}_fold{fold_num}.csv"

            img_path   = os.path.join(mode_dir, img_fname)
            slide_path = os.path.join(mode_dir, slide_fname)

            _write_image_csv(img_path,   i_data)
            _write_slide_csv(slide_path, s_data)
            image_paths.append(os.path.abspath(img_path))

        written[mode] = image_paths

    total_files = sum(len(v) for v in written.values())
    print(f"\nDone. {total_files} image-split CSV(s) written "
          f"across {len(written)} mode(s).")
    print(f"Each mode folder also contains a slide-assignment CSV per fold.")
    return written

# ── CLI ────────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(
        description=("Generate per-fold train/val/test split CSVs with zero data leakage.\n"
                    "Splitting happens at the SLIDE level — all tiles from the same\n"
                    "slide always land in the same split." ),
                    formatter_class=argparse.RawDescriptionHelpFormatter,)
    p.add_argument("--images_dir",   type=str, required=True, help="Root folder to scan (searched recursively)")
    p.add_argument("--output_dir",   type=str, default="splits", help="Root output folder (default: splits/)")
    p.add_argument("--dataset_name", type=str, required=True, help="Label used in CSV filenames")
    p.add_argument("--folds",        type=int,   default=1, help="Number of folds (default: 1)")
    p.add_argument("--val_split",    type=float, default=0.1, help="Fraction of SLIDES for validation (default: 0.1)")
    p.add_argument("--test_split",   type=float, default=0.2, help="Fraction of SLIDES for test (default: 0.2)")
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
