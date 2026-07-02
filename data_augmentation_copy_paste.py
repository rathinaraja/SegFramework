"""
augment_mask_patches.py
-----------------------
Augments mask–patch pairs by copying white-region tiles into black regions
when a mask has >60% black pixels.

Non-overlap guarantee
---------------------
When --copies_per_mask > 1, both source (white) and destination (black) tile
placements are kept strictly non-overlapping across all copies in the same
image.  If fewer than the requested number of non-overlapping placements are
available, the maximum possible non-overlapping copies are performed and a
warning is printed.

The non-overlap tracking works by maintaining two boolean occupancy grids
(one for source positions, one for destination positions).  After each
placement the corresponding n×n footprint is marked as occupied, and those
positions are removed from the candidate pool before the next draw.

Usage:
python data_augmentation_copy_paste.py \
    --masks_dir   /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/traininng/masks/ \
    --patches_dir /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/traininng/patches/ \
    --output_dir  /data_64T_3/Raja/CDH1/src_segmentation/dataset_augmentation/augmented_copy_paste \
    --tile_size   64 \
    --black_thresh 50 \
    --apply_pct   100 \
    --copies_per_mask 2 \
    --seed 42

Arguments:
    --masks_dir       : Path to folder containing binary mask images
    --patches_dir     : Path to folder with corresponding patch images
    --output_dir      : Where augmented mask+patch pairs are saved
    --tile_size       : n for the n×n tile (default: 64)
    --black_thresh    : Minimum % of black pixels to trigger augmentation (default: 60)
    --apply_pct       : % of qualifying masks to augment (0–100, default: 100)
    --copies_per_mask : Max white→black copy operations per mask (default: 1)
    --seed            : Random seed for reproducibility (default: 42)
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ─────────────────────────── helpers ────────────────────────────────────────

def load_mask_as_binary(mask_path: Path) -> np.ndarray:
    """
    Load a mask and return a boolean array: True = white pixel, False = black.
    Handles grayscale and RGB masks; threshold at 127.
    """
    img = Image.open(mask_path).convert("L")
    return np.array(img) > 127          # True → white, False → black

def black_pixel_pct(binary_mask: np.ndarray) -> float:
    """Return % of black (False) pixels in a binary mask."""
    return 100.0 * np.sum(~binary_mask) / binary_mask.size

def _build_integral(region: np.ndarray) -> np.ndarray:
    """2-D prefix-sum (integral image) of a boolean region."""
    return region.astype(np.int32).cumsum(axis=0).cumsum(axis=1)

def _rect_sum(integral: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> int:
    """Sum of region[r1:r2+1, c1:c2+1] using the pre-built integral image."""
    s = integral[r2, c2]
    if r1 > 0:            s -= integral[r1 - 1, c2]
    if c1 > 0:            s -= integral[r2,     c1 - 1]
    if r1 > 0 and c1 > 0: s += integral[r1 - 1, c1 - 1]
    return s

def find_valid_tile_positions(region: np.ndarray, tile_size: int) -> list[tuple[int, int]]:
    """
    Return (row, col) top-left corners of all n×n tiles that lie entirely
    inside *region* (i.e. every pixel in the tile belongs to the region).
    Uses an integral image for O(1) per-tile checks.
    """
    h, w = region.shape
    n = tile_size
    if h < n or w < n:
        return []

    integral = _build_integral(region)
    positions = []
    for r in range(h - n + 1):
        for c in range(w - n + 1):
            if _rect_sum(integral, r, c, r + n - 1, c + n - 1) == n * n:
                positions.append((r, c))
    return positions

def _mark_occupied(occupied: np.ndarray, r: int, c: int, tile_size: int) -> None:
    """Mark the n×n footprint starting at (r, c) as occupied (in-place)."""
    occupied[r: r + tile_size, c: c + tile_size] = True

def _filter_non_overlapping(positions: list[tuple[int, int]], occupied: np.ndarray, tile_size: int,) -> list[tuple[int, int]]:
    """
    Return only those positions whose n×n footprint does not overlap with
    any already-occupied cell.
    Uses an integral image over the occupied grid for fast filtering.
    """
    if not positions:
        return []
    integral = _build_integral(occupied)
    n = tile_size
    return [
        (r, c) for r, c in positions
        if _rect_sum(integral, r, c, r + n - 1, c + n - 1) == 0
    ]

# ─────────────────────────── core augmentation ───────────────────────────────

def augment_pair(mask_path: Path,patch_path: Path,out_mask_path: Path,out_patch_path: Path,tile_size: int,copies: int,rng: random.Random,) -> int:
    """
    Perform non-overlapping augmentation on one mask–patch pair.

    For each copy:
      • A source tile is drawn at random from white positions not yet used
        as a source in this image.
      • A destination tile is drawn at random from black positions not yet
        used as a destination in this image.
      • Both footprints are marked occupied so no future copy overlaps them.

    If fewer than *copies* non-overlapping placements are possible, as many
    as possible are performed.

    Returns the number of copies actually placed (0 if nothing could be done).
    """
    binary_mask = load_mask_as_binary(mask_path)
    mask_img    = Image.open(mask_path).convert("RGB")
    patch_img   = Image.open(patch_path).convert("RGB")

    mask_arr  = np.array(mask_img)
    patch_arr = np.array(patch_img)
    h, w      = binary_mask.shape

    # All candidate positions (computed once)
    all_white_pos = find_valid_tile_positions(binary_mask,  tile_size)
    all_black_pos = find_valid_tile_positions(~binary_mask, tile_size)

    if not all_white_pos or not all_black_pos:
        print(f"  [SKIP] {mask_path.name}: no tile-sized white or black regions.")
        return 0

    # Occupancy grids — track used footprints separately for src and dst
    src_occupied = np.zeros((h, w), dtype=bool)
    dst_occupied = np.zeros((h, w), dtype=bool)

    aug_mask  = mask_arr.copy()
    aug_patch = patch_arr.copy()
    placed    = 0

    for i in range(copies):
        # Filter to positions whose footprint is still free
        free_white = _filter_non_overlapping(all_white_pos, src_occupied, tile_size)
        free_black = _filter_non_overlapping(all_black_pos, dst_occupied, tile_size)

        if not free_white or not free_black:
            print(
                f"  [WARN] {mask_path.name}: only {placed}/{copies} non-overlapping "
                f"copies possible — stopping early."
            )
            break

        src_r, src_c = rng.choice(free_white)
        dst_r, dst_c = rng.choice(free_black)

        # Apply the copy to both mask and patch
        n = tile_size
        aug_mask [dst_r:dst_r+n, dst_c:dst_c+n] = mask_arr [src_r:src_r+n, src_c:src_c+n]
        aug_patch[dst_r:dst_r+n, dst_c:dst_c+n] = patch_arr[src_r:src_r+n, src_c:src_c+n]

        # Mark footprints as occupied
        _mark_occupied(src_occupied, src_r, src_c, tile_size)
        _mark_occupied(dst_occupied, dst_r, dst_c, tile_size)

        placed += 1
        print(
            f"    Copy {placed}/{copies}: "
            f"white({src_r},{src_c}) → black({dst_r},{dst_c})"
        )

    if placed > 0:
        Image.fromarray(aug_mask).save(out_mask_path)
        Image.fromarray(aug_patch).save(out_patch_path)

    return placed

# ─────────────────────────── main ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Augment mask–patch pairs by copying white-region tiles into black "
            "regions. Multiple copies per image are guaranteed non-overlapping; "
            "if the image cannot accommodate the requested number, the maximum "
            "possible non-overlapping copies are placed instead."
        )
    )
    parser.add_argument("--masks_dir",       type=str, required=True)
    parser.add_argument("--patches_dir",     type=str, required=True)
    parser.add_argument("--output_dir",      type=str, default="augmented")
    parser.add_argument("--tile_size",       type=int, default=64, help="n for n×n tile (default: 64)")
    parser.add_argument("--black_thresh",    type=float, default=60.0, help="Min %% black pixels to qualify a mask (default: 60)")
    parser.add_argument("--apply_pct",       type=float, default=100.0, help="%% of qualifying masks to actually augment (default: 100)")
    parser.add_argument("--copies_per_mask", type=int, default=1, help="Max non-overlapping white→black copies per mask (default: 1)")
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    masks_dir   = Path(args.masks_dir)
    patches_dir = Path(args.patches_dir)
    output_dir  = Path(args.output_dir)

    out_masks_dir   = output_dir / "masks"
    out_patches_dir = output_dir / "patches"
    out_masks_dir.mkdir(parents=True, exist_ok=True)
    out_patches_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    mask_files = sorted(f for f in masks_dir.iterdir() if f.suffix.lower() in exts)

    if not mask_files:
        print(f"No image files found in {masks_dir}")
        sys.exit(1)

    print(f"Found {len(mask_files)} mask(s) in '{masks_dir}'")
    print(
        f"Settings: tile={args.tile_size}px | black_thresh={args.black_thresh}% "
        f"| apply_pct={args.apply_pct}% | copies={args.copies_per_mask} "
        f"| seed={args.seed}\n"
    )

    # ── Step 1: find qualifying masks ──────────────────────────────────────
    qualifying        = []
    skipped_no_patch  = 0
    skipped_low_black = 0

    for mask_path in mask_files:
        stem = mask_path.stem

        patch_path = None
        for ext in exts:
            candidate = patches_dir / (stem + ext)
            if candidate.exists():
                patch_path = candidate
                break

        if patch_path is None:
            print(f"[WARN] No matching patch for '{mask_path.name}' – skipping.")
            skipped_no_patch += 1
            continue

        binary_mask = load_mask_as_binary(mask_path)
        bpct = black_pixel_pct(binary_mask)

        if bpct >= args.black_thresh:
            qualifying.append((mask_path, patch_path, bpct))
        else:
            skipped_low_black += 1

    print(f"Qualifying masks (≥{args.black_thresh}% black): {len(qualifying)}")
    print(f"Skipped (no patch):    {skipped_no_patch}")
    print(f"Skipped (< threshold): {skipped_low_black}\n")

    if not qualifying:
        print("Nothing to augment.")
        sys.exit(0)

    # ── Step 2: subsample by apply_pct ────────────────────────────────────
    n_to_apply = max(1, round(len(qualifying) * args.apply_pct / 100.0))
    selected   = rng.sample(qualifying, min(n_to_apply, len(qualifying)))

    print(
        f"Augmenting {len(selected)} / {len(qualifying)} qualifying masks "
        f"({args.apply_pct:.1f}% requested)\n"
    )

    # ── Step 3: augment ────────────────────────────────────────────────────
    total_placed = 0
    success      = 0

    for mask_path, patch_path, bpct in selected:
        stem           = mask_path.stem
        out_mask_path  = out_masks_dir   / f"{mask_path.stem}_augmented{mask_path.suffix}"
        out_patch_path = out_patches_dir / f"{patch_path.stem}_augmented{patch_path.suffix}"

        print(f"[{stem}]  black={bpct:.1f}%")
        placed = augment_pair(
            mask_path, patch_path,
            out_mask_path, out_patch_path,
            tile_size=args.tile_size,
            copies=args.copies_per_mask,
            rng=rng,
        )
        if placed > 0:
            success      += 1
            total_placed += placed

    print(
        f"\n✓ Done. {success}/{len(selected)} pairs augmented "
        f"({total_placed} total tile copies placed) → '{output_dir}'"
    )

if __name__ == "__main__":
    main()