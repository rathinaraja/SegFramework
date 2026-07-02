import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import json

# --- Configuration & Paths ---
TRAIN_PATCHES_DIR = "/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/traininng/patches"
TRAIN_MASKS_DIR   = "/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/traininng/masks"
TEST_PATCHES_DIR  = "/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/test/patches"

OUT_ROOT = Path("/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_raw/Dataset101_SRC")

images_tr = OUT_ROOT / "imagesTr"
labels_tr = OUT_ROOT / "labelsTr"
images_ts = OUT_ROOT / "imagesTs"

# Create all necessary directories
images_tr.mkdir(parents=True, exist_ok=True)
labels_tr.mkdir(parents=True, exist_ok=True)
images_ts.mkdir(parents=True, exist_ok=True)

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# --- 1. Process Training Data ---
print("Processing Training Data...")
train_patches = sorted([p for p in Path(TRAIN_PATCHES_DIR).iterdir() 
                        if p.suffix.lower() in IMG_EXT])

case_ids = []
for patch_path in train_patches:
    # Find matching mask
    mask_path = None
    for ext in IMG_EXT:
        candidate = Path(TRAIN_MASKS_DIR) / f"{patch_path.stem}{ext}"
        if candidate.exists():
            mask_path = candidate
            break
            
    if mask_path is None:
        print(f"SKIP — no mask for {patch_path.name}")
        continue

    case_id = patch_path.stem
    case_ids.append(case_id)

    # Copy image as {case_id}_0000.png (nnUNet channel naming)
    shutil.copy2(patch_path, images_tr / f"{case_id}_0000.png")

    # Convert mask: 255->1, 0->0 and save as {case_id}.png
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = (mask / 255).round().astype(np.uint8)   
    Image.fromarray(mask).save(labels_tr / f"{case_id}.png")

print(f"Converted {len(case_ids)} training cases.")

# --- 2. Process Testing Data ---
print("\nProcessing Testing Data...")
test_patches = sorted([p for p in Path(TEST_PATCHES_DIR).iterdir() 
                       if p.suffix.lower() in IMG_EXT])

test_count = 0
for patch_path in test_patches:
    case_id = patch_path.stem
    
    # Copy test image as {case_id}_0000.png (Masks are not needed for testing images)
    shutil.copy2(patch_path, images_ts / f"{case_id}_0000.png")
    test_count += 1

print(f"Converted {test_count} testing cases.")

# --- 3. Write dataset.json ---
print("\nWriting dataset.json...")
dataset_json = {
    "channel_names": {"0": "RGB"},   # single PNG with 3 channels
    "labels": {
        "background": 0,
        "foreground": 1
    },
    "numTraining": len(case_ids),
    "file_ending": ".png"
}

with open(OUT_ROOT / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=2)

print(f"Dataset completely saved to: {OUT_ROOT}")