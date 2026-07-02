"""
nnunet_prepare.py
------------------
Step 1: Converts your patches/masks folders to nnU-Net format
        AND generates splits_final.json from your CSV fold files.

Usage:
    python nnunet_prepare.py \
        --patches_dir   /data_64T_3/Raja/Alex_project/dataset/patches \
        --masks_dir     /data_64T_3/Raja/Alex_project/dataset/masks \
        --splits_dir    /input_path_split_files/ \
        --dataset_id    101 \
        --dataset_name  SRC

After running this:
    nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def find_file(stem, directory):
    for ext in IMG_EXT:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
        p = directory / f"{Path(stem).stem}{ext}"
        if p.exists():
            return p
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patches_dir",   required=True)
    p.add_argument("--masks_dir",     required=True)
    p.add_argument("--splits_dir",    required=True)
    p.add_argument("--dataset_id",    type=int, default=101)
    p.add_argument("--dataset_name",  default="WSIPatches")
    return p.parse_args()


def main():
    args       = parse_args()
    patches_dir = Path(args.patches_dir)
    masks_dir   = Path(args.masks_dir)
    splits_dir  = Path(args.splits_dir)

    # ── Set up nnUNet dataset folder ──────────────────────────────────────────
    raw_root  = Path(os.environ["nnUNet_raw"])
    ds_folder = raw_root / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_tr = ds_folder / "imagesTr"
    labels_tr = ds_folder / "labelsTr"
    images_ts = ds_folder / "imagesTs"   # all test images across folds go here
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)

    # ── Read all CSV files and collect unique case IDs ────────────────────────
    csv_files = sorted(splits_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files: print(f"  {f.name}")

    all_train, all_val, all_test = set(), set(), set()
    fold_data = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        train = df["train"].dropna().astype(str).tolist() if "train" in df.columns else []
        val   = df["val"].dropna().astype(str).tolist()   if "val"   in df.columns else []
        test  = df["test"].dropna().astype(str).tolist()  if "test"  in df.columns else []
        all_train.update(train); all_val.update(val); all_test.update(test)
        fold_data.append({"csv": csv_path.name, "train": train, "val": val, "test": test})

    # All unique cases
    all_cases = all_train | all_val | all_test
    print(f"\nTotal unique cases: {len(all_cases)}")
    print(f"  Train-only cases : {len(all_train - all_test)}")
    print(f"  Test cases       : {len(all_test)}")

    # ── Copy images and masks ─────────────────────────────────────────────────
    skipped = []
    for stem in sorted(all_cases):
        stem_clean = Path(stem).stem   # strip any extension in CSV value
        img_path   = find_file(stem_clean, patches_dir)
        mask_path  = find_file(stem_clean, masks_dir)

        if img_path is None or mask_path is None:
            skipped.append(stem_clean)
            continue

        # Train/val cases → imagesTr / labelsTr
        if stem_clean in (all_train | all_val):
            dst_img  = images_tr / f"{stem_clean}_0000.png"
            dst_mask = labels_tr / f"{stem_clean}.png"
            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)
            if not dst_mask.exists():
                mask_arr = np.array(Image.open(mask_path).convert("L"))
                mask_arr = (mask_arr / 255).round().astype(np.uint8)
                Image.fromarray(mask_arr).save(dst_mask)

        # Test cases → imagesTs (nnUNet won't train on these)
        if stem_clean in all_test:
            dst_test = images_ts / f"{stem_clean}_0000.png"
            if not dst_test.exists():
                shutil.copy2(img_path, dst_test)

    train_val_cases = sorted(all_train | all_val)
    print(f"\nCopied {len(train_val_cases)} train/val images to imagesTr")
    print(f"Copied {len(all_test)} test images to imagesTs")
    if skipped:
        print(f"SKIPPED {len(skipped)}: {skipped[:5]}")

    # ── Generate dataset.json ─────────────────────────────────────────────────
    dataset_json = {
        "channel_names": {"0": "RGB"},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": len(train_val_cases),
        "file_ending": ".png"
    }
    with open(ds_folder / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"\ndataset.json written: {ds_folder / 'dataset.json'}")

    # ── Generate splits_final.json ────────────────────────────────────────────
    # Format: [{"train": [case_ids], "val": [case_ids]}, ...]
    # Note: test split is NOT in splits_final.json — nnUNet only uses train/val.
    # Test evaluation is done separately in step 3 (nnunet_evaluate.py).
    splits_json = []
    for fd in fold_data:
        train_ids = [Path(s).stem for s in fd["train"]]
        val_ids   = [Path(s).stem for s in fd["val"]]
        
        # ONLY append if the fold actually has training or validation data
        if train_ids or val_ids:
            splits_json.append({"train": train_ids, "val": val_ids})

    if not splits_json:
        print("WARNING: No valid training/val splits found in your CSV files!")
        
    # Write the splits_final.json file and define splits_out
    splits_out = ds_folder / "splits_final.json"
    with open(splits_out, "w") as f:
        json.dump(splits_json, f, indent=2)
    print(f"splits_final.json written: {splits_out}")

    # splits_final.json goes in nnUNet_preprocessed AFTER plan_and_preprocess
    # ── Generate test_splits.json ────────────────────────────────────────────
    # Save test case IDs per fold for evaluation later
    test_split = {}
    valid_fold_idx = 0  # Initialize a continuous counter starting at 0
    
    for fd in fold_data:
        test_ids = [Path(s).stem for s in fd["test"]]
        
        # ONLY process if the fold actually contains test images
        if test_ids:
            # Assign to the sequential counter (fold0, fold1, etc.)
            test_split[f"fold{valid_fold_idx}"] = test_ids
            valid_fold_idx += 1  # Increment for the next valid fold
            
    test_out = ds_folder / "test_splits.json"
    with open(test_out, "w") as f:
        json.dump(test_split, f, indent=2)
        
    print(f"test_splits.json written : {test_out}")

    print(f"\n{'='*60}")
    print("Next step:")
    print(f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity")
    print(f"  cp {splits_out} $nnUNet_preprocessed/Dataset{args.dataset_id:03d}_{args.dataset_name}/splits_final.json")


if __name__ == "__main__":
    main()
