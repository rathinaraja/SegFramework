"""
visualize.py
-------------
For each input patch + ground truth mask pair, saves one output image
containing 4 panels side by side:

    Input Patch | Ground Truth | Predicted Mask | Overlay (GT vs Pred)

One PNG is saved per input patch.

Usage:
    python visualization_generation.py --config configs/segnet.yaml \
                        --checkpoint /home/rajaj/Project/Alex_project/seg_framework/logs/segnet_dataset/20260605_234346/fold_1/checkpoints/best_model.pth \
                        --images_dir /data_55T_2/Raja/SEGMENTATION/test/patches \
                        --masks_dir  /data_55T_2/Raja/SEGMENTATION/test/masks \
                        --output_dir outputs/segnet \
                        --device cuda:1  

    # Limit to N samples (random pick)
    python visualize.py ... --n 10 --random

    # First N samples in sorted order
    python visualize.py ... --n 10

    # Override device
    python visualize.py ... --device cuda:1


python visualization_generation.py --config configs/attention_unet.yaml \
                        --checkpoint /home/rajaj/Project/Alex_project/seg_framework/logs/attention_unet_dataset/20260605_234346/fold_1/checkpoints/best_model.pth \
                        --images_dir /data_55T_2/Raja/SEGMENTATION/test/patches \
                        --masks_dir  /data_55T_2/Raja/SEGMENTATION/test/masks \
                        --output_dir /data_55T_2/Raja/SEGMENTATION/outputs/attention_unet \
                        --device cuda:1  

python visualization_generation.py --config configs/nnunet.yaml \
                        --checkpoint /home/rajaj/Project/Alex_project/seg_framework/logs/nnunet_dataset/20260605_234346/fold_5/checkpoints/best_model.pth \
                        --images_dir /data_55T_2/Raja/SEGMENTATION/test/patches \
                        --masks_dir  /data_55T_2/Raja/SEGMENTATION/test/masks \
                        --output_dir /data_55T_2/Raja/SEGMENTATION/outputs/nnunet \
                        --device cuda:1  

python visualization_generation.py --config configs/segformer.yaml \
                        --checkpoint /home/rajaj/Project/Alex_project/seg_framework/logs/segformer_dataset/20260605_234346/fold_4/checkpoints/best_model.pth \
                        --images_dir /data_55T_2/Raja/SEGMENTATION/test/patches \
                        --masks_dir  /data_55T_2/Raja/SEGMENTATION/test/masks \
                        --output_dir /data_55T_2/Raja/SEGMENTATION/outputs/segformer \
                        --device cuda:1  

python visualization_generation.py --config configs/segnet.yaml \
                        --checkpoint /home/rajaj/Project/Alex_project/seg_framework/logs/segnet_dataset/20260605_234346/fold_5/checkpoints/best_model.pth \
                        --images_dir /data_55T_2/Raja/SEGMENTATION/test/patches \
                        --masks_dir  /data_55T_2/Raja/SEGMENTATION/test/masks \
                        --output_dir /data_55T_2/Raja/SEGMENTATION/outputs/segnet \
                        --device cuda:1  

python visualization_generation.py --config configs/swinunet.yaml \
                        --checkpoint /home/rajaj/Project/Alex_project/seg_framework/logs/swinunet_dataset/20260605_234346/fold_5/checkpoints/best_model.pth \
                        --images_dir /data_55T_2/Raja/SEGMENTATION/test/patches \
                        --masks_dir  /data_55T_2/Raja/SEGMENTATION/test/masks \
                        --output_dir /data_55T_2/Raja/SEGMENTATION/outputs/swinunet \
                        --device cuda:1  

python visualization_generation.py --config configs/transunet.yaml \
                        --checkpoint /home/rajaj/Project/Alex_project/seg_framework/logs/transunet_dataset/20260605_234346/fold_1/checkpoints/best_model.pth \
                        --images_dir /data_55T_2/Raja/SEGMENTATION/test/patches \
                        --masks_dir  /data_55T_2/Raja/SEGMENTATION/test/masks \
                        --output_dir /data_55T_2/Raja/SEGMENTATION/outputs/transunet \
                        --device cuda:1  

python visualization_generation.py --config configs/unet.yaml \
                        --checkpoint /home/rajaj/Project/Alex_project/seg_framework/logs/unet_dataset/20260605_234346/fold_5/checkpoints/best_model.pth \
                        --images_dir /data_55T_2/Raja/SEGMENTATION/test/patches \
                        --masks_dir  /data_55T_2/Raja/SEGMENTATION/test/masks \
                        --output_dir /data_55T_2/Raja/SEGMENTATION/outputs/unet \
                        --device cuda:1  

python visualization_generation.py --config configs/unetpp.yaml \
                        --checkpoint /home/rajaj/Project/Alex_project/seg_framework/logs/unetpp_dataset/20260605_234346/fold_5/checkpoints/best_model.pth \
                        --images_dir /data_55T_2/Raja/SEGMENTATION/test/patches \
                        --masks_dir  /data_55T_2/Raja/SEGMENTATION/test/masks \
                        --output_dir /data_55T_2/Raja/SEGMENTATION/outputs/unetpp \
                        --device cuda:1  
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.ndimage import binary_erosion
from torch.amp import autocast
import torchvision.transforms.functional as TF

from utils.config import load_config
from modules      import get_model

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_COLORS = [
    (0,   0,   0),    # class 0 — background (black)
    (255, 255, 255),  # class 1 — foreground (white)
    (255, 0,   0),    # class 2 — red
    (0,   255, 0),    # class 3 — green
    (0,   0,   255),  # class 4 — blue
]
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ── Helpers ────────────────────────────────────────────────────────────────────
def colorize(mask, colors):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        rgb[mask == i] = c
    return rgb

def preprocess(path, img_size, device):
    orig   = Image.open(path).convert("RGB").resize(
                 (img_size[1], img_size[0]), Image.BILINEAR)
    tensor = TF.normalize(TF.to_tensor(orig), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return np.array(orig), tensor.unsqueeze(0).to(device)

def load_gt(path, img_size):
    gt = Image.open(path).convert("L").resize((img_size[1], img_size[0]), Image.NEAREST)
    return (np.array(gt, dtype=np.float32) / 255).round().astype(np.int64)

def iou_dice(pred, gt, n_cls, eps=1e-6):
    ious, dices = [], []
    for c in range(n_cls):
        p, t  = (pred == c), (gt == c)
        inter = (p & t).sum()
        union = (p | t).sum()
        denom = p.sum() + t.sum()
        if union > 0:  ious.append(inter / (union + eps))
        if denom > 0: dices.append(2 * inter / (denom + eps))
    return (float(np.mean(ious))  if ious  else 0.0,
            float(np.mean(dices)) if dices else 0.0)

def find_mask(masks_dir, img_stem):
    for ext in IMG_EXT:
        p = masks_dir / f"{img_stem}{ext}"
        if p.exists():
            return p
    return None

# ── Per-image 4-panel save ─────────────────────────────────────────────────────
def draw_boundary(mask_rgb: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Draw black boundary lines around each class region."""
    result = mask_rgb.copy()
    for cls in np.unique(pred):
        region   = (pred == cls)
        eroded   = binary_erosion(region, iterations=1)
        boundary = region & ~eroded
        result[boundary] = [0, 0, 0]
    return result

def save_panel(orig, gt, pred, iou, dice, n_cls, save_path, model_tag, fname):
    gt_rgb   = colorize(gt,   CLASS_COLORS)
    pred_rgb = colorize(pred, CLASS_COLORS)
    # Black boundaries on GT and predicted masks
    gt_rgb   = draw_boundary(gt_rgb,   gt)
    pred_rgb = draw_boundary(pred_rgb, pred)
    # Overlay 1: GT blended over input
    overlay_gt   = (orig * 0.5 + gt_rgb   * 0.5).astype(np.uint8)
    # Overlay 2: Prediction blended over input
    overlay_pred = (orig * 0.5 + pred_rgb * 0.5).astype(np.uint8)
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    axes[0].imshow(orig);         axes[0].set_title("Input Patch",                          fontsize=11, fontweight="bold")
    axes[1].imshow(gt_rgb);       axes[1].set_title("Ground Truth",                         fontsize=11, fontweight="bold")
    axes[2].imshow(overlay_gt);   axes[2].set_title("Overlay\n(Ground Truth on Input)",     fontsize=11, fontweight="bold")
    axes[3].imshow(pred_rgb);     axes[3].set_title( f"Predicted Mask\nIoU={iou:.4f}   Dice={dice:.4f}",  fontsize=11, fontweight="bold")
    axes[4].imshow(overlay_pred); axes[4].set_title("Overlay\n(Predicted Mask on Input)",   fontsize=11, fontweight="bold")

    for ax in axes:
        ax.axis("off")

    # Class legend
    legend_patches = [mpatches.Patch(color=[c/255 for c in CLASS_COLORS[i]], label=f"Class {i}") for i in range(n_cls)]
    fig.legend(handles=legend_patches, loc="lower center", ncol=n_cls, fontsize=9, bbox_to_anchor=(0.5, -0.04), frameon=True)

    fig.suptitle(f"{model_tag}   |   {fname}", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Save one 4-panel viz image per patch: " "Input | GT | Prediction | Overlay")
    p.add_argument("--config",      required=True,  help="Path to YAML config")
    p.add_argument("--checkpoint",  required=True,  help="Path to .pth checkpoint")
    p.add_argument("--images_dir",  required=True,  help="Input patches folder")
    p.add_argument("--masks_dir",   required=True,  help="Ground truth masks folder")
    p.add_argument("--output_dir",  default="outputs/viz", help="Output folder for viz images (default: outputs/viz)")
    p.add_argument("--n",           type=int, default=None, help="Process only N patches (default: all)")
    p.add_argument("--random",      action="store_true", help="Randomly pick N patches instead of first N")
    p.add_argument("--device",      default=None, help="Device override: cuda, cuda:0, cpu")
    return p.parse_args()

def main():
    args = parse_args()
    # ── Config + model ────────────────────────────────────────────────────────
    cfg      = load_config(args.config)
    device   = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    img_size = tuple(cfg["dataset"].get("img_size", [512, 512]))
    amp      = cfg["training"].get("amp", True)
    n_cls    = cfg["model"]["n_classes"]
    os.makedirs(args.output_dir, exist_ok=True)
    model = get_model(cfg)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    model_tag = f"{cfg['model']['name'].upper()} | Epoch {ckpt.get('epoch','?')}"
    print(f"Model     : {model_tag}")
    print(f"Device    : {device}")
    print(f"img_size  : {img_size}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'─'*60}")

    # ── Collect image paths ───────────────────────────────────────────────────
    images_dir = Path(args.images_dir)
    masks_dir  = Path(args.masks_dir)
    all_imgs   = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT])

    if args.n:
        all_imgs = (random.sample(all_imgs, min(args.n, len(all_imgs))) if args.random else all_imgs[:args.n])

    print(f"Total to process: {len(all_imgs)} patches\n")

    # ── Process ───────────────────────────────────────────────────────────────
    skipped    = 0
    ious, dices = [], []

    with torch.no_grad():
        for i, img_path in enumerate(all_imgs, 1):
            mask_path = find_mask(masks_dir, img_path.stem)
            if mask_path is None:
                print(f"  [{i:>4}/{len(all_imgs)}] SKIP — no mask: {img_path.name}")
                skipped += 1
                continue

            orig, tensor = preprocess(img_path, img_size, device)
            gt           = load_gt(mask_path, img_size)

            with autocast('cuda', enabled=amp and device.type == 'cuda'):
                logits = model(tensor)
            pred = logits.argmax(1).squeeze(0).cpu().numpy()

            iou, dice = iou_dice(pred, gt, n_cls)
            ious.append(iou)
            dices.append(dice)

            out_path = os.path.join(args.output_dir, f"{img_path.stem}_viz.png")
            save_panel(orig, gt, pred, iou, dice, n_cls, out_path, model_tag, img_path.name)
            print(f"  [{i:>4}/{len(all_imgs)}] {img_path.name:<40} " f"IoU={iou:.4f}  Dice={dice:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Saved   : {len(ious)} images  |  Skipped: {skipped}")
    if ious:
        print(f"Mean IoU : {np.mean(ious):.4f}")
        print(f"Mean Dice: {np.mean(dices):.4f}")
    print(f"Output   : {args.output_dir}")

if __name__ == "__main__":
    main()