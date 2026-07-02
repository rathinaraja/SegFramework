"""
nnunet_evaluate.py
-------------------
Usage:
    python nnunet_evaluate.py \
        --dataset_id    101 \
        --dataset_name  SRC \
        --masks_dir     /path/to/masks \
        --splits_dir    /path/to/fold.csv  OR  /path/to/csv_folder/ \
        --output_dir    /path/to/results \
        --device        cuda:1 \
        --checkpoint    /path/to/checkpoint_best.pth \
        --fold          fold0
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUMMARY_COLS = [
    "fold", "seed",
    "train_loss", "train_acc", "train_iou", "train_dice",
    "val_loss",   "val_acc",   "val_iou",   "val_dice",
    "test_loss",  "test_acc",  "test_iou",  "test_dice",
]
CLASS_COLORS = [(0, 0, 0), (255, 255, 255)]


# ── Image helpers ──────────────────────────────────────────────────────────────

def load_pred(pred_path):
    arr = np.array(Image.open(pred_path).convert("L"))
    if arr.max() > 1:
        arr = (arr / 255).round().astype(np.uint8)
    return arr


def load_gt(gt_path):
    arr = np.array(Image.open(gt_path).convert("L"))
    if arr.max() > 1:
        arr = (arr / 255).round().astype(np.uint8)
    return arr


def load_patch(patch_path):
    return np.array(Image.open(patch_path).convert("RGB"))


def colorize(mask, colors=CLASS_COLORS):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        rgb[mask == i] = c
    return rgb


def find_file(stem, directory):
    for ext in IMG_EXT:
        p = directory / (stem + ext)
        if p.exists():
            return p
    return None


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(pred, gt, eps=1e-6):
    pred  = (pred > 0).astype(np.int64)
    gt    = (gt   > 0).astype(np.int64)
    acc   = (pred == gt).mean()
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    iou   = inter / (union + eps) if union > 0 else 1.0
    denom = pred.sum() + gt.sum()
    dice  = 2 * inter / (denom + eps) if denom > 0 else 1.0
    return float(acc), float(iou), float(dice)


# ── Visualisation ──────────────────────────────────────────────────────────────

def save_viz(patch_arr, gt_arr, pred_arr, iou, dice, acc, save_path, case_id):
    gt_rgb       = colorize(gt_arr)
    pred_rgb     = colorize(pred_arr)
    overlay_gt   = (patch_arr * 0.5 + gt_rgb   * 0.5).astype(np.uint8)
    overlay_pred = (patch_arr * 0.5 + pred_rgb * 0.5).astype(np.uint8)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(patch_arr);    axes[0].set_title("Input Patch",         fontsize=11, fontweight="bold")
    axes[1].imshow(gt_rgb);       axes[1].set_title("Ground Truth",        fontsize=11, fontweight="bold")
    axes[2].imshow(overlay_gt);   axes[2].set_title("Overlay\n(GT on Input)", fontsize=11, fontweight="bold")
    axes[3].imshow(pred_rgb);     axes[3].set_title(
        "Predicted Mask\nIoU={:.4f}  Dice={:.4f}".format(iou, dice),       fontsize=11, fontweight="bold")
    axes[4].imshow(overlay_pred); axes[4].set_title("Overlay\n(Pred on Input)", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.axis("off")

    legend_patches = [
        mpatches.Patch(color=[c/255 for c in CLASS_COLORS[0]], label="Background"),
        mpatches.Patch(color=[c/255 for c in CLASS_COLORS[1]], label="Foreground",
                       linewidth=1, edgecolor="gray"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.04), frameon=True)
    fig.suptitle(case_id, fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=120, bbox_inches="tight")
    plt.close()


# ── CSV helpers ────────────────────────────────────────────────────────────────

def is_valid_split_csv(csv_path):
    try:
        import pandas as pd
        df   = pd.read_csv(csv_path, nrows=0)
        cols = {c.strip().lower() for c in df.columns}
        return {"train", "val", "test"}.issubset(cols)
    except Exception:
        return False


# ── nnUNet log parsing ─────────────────────────────────────────────────────────

def parse_nnunet_training_log(fold_result_dir):
    log_files  = sorted(fold_result_dir.glob("training_log_*.txt"))
    if not log_files:
        return {"train_loss": None, "train_acc": None,
                "train_iou":  None, "train_dice": None}
    log_text   = log_files[-1].read_text()
    train_loss = None
    for line in log_text.splitlines():
        m = re.search(r'train[_ ]loss[:\s]+(-?[\d.]+)', line, re.IGNORECASE)
        if m:
            train_loss = abs(float(m.group(1)))
    train_dice = train_loss
    return {
        "train_loss": round(1 - train_dice, 4) if train_dice else None,
        "train_acc":  None,
        "train_iou":  None,
        "train_dice": round(train_dice, 4) if train_dice else None,
    }


def parse_nnunet_val_summary(fold_result_dir):
    summary_path = fold_result_dir / "validation" / "summary.json"
    if not summary_path.exists():
        return {"val_loss": None, "val_acc": None,
                "val_iou":  None, "val_dice": None}
    with open(summary_path) as f:
        summary = json.load(f)
    mean = summary.get("mean", {})
    cls1 = mean.get("1", mean.get("foreground", {}))
    dice = cls1.get("Dice", None)
    iou  = cls1.get("IoU",  None)
    return {
        "val_loss":  round(1 - dice, 4) if dice else None,
        "val_acc":   None,
        "val_iou":   round(iou,  4) if iou  else None,
        "val_dice":  round(dice, 4) if dice else None,
    }


# ── Inference ──────────────────────────────────────────────────────────────────

def run_test_inference(dataset_id, config, fold_idx,
                       test_images_dir, pred_output_dir,
                       device, checkpoint_path=None):
    pred_output_dir.mkdir(parents=True, exist_ok=True)
    gpu_id = device.replace("cuda:", "")
    env    = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    cmd = [
        "nnUNetv2_predict",
        "-i", str(test_images_dir),
        "-o", str(pred_output_dir),
        "-d", str(dataset_id),
        "-c", config,
        "-f", str(fold_idx),
    ]
    if checkpoint_path:
        cmd += ["-chk", str(checkpoint_path)]
    print("    Running: " + " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print("    WARNING: nnUNetv2_predict returned code " + str(result.returncode))


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_fold(pred_dir, test_images_dir, masks_dir, test_case_ids, viz_dir):
    viz_dir.mkdir(parents=True, exist_ok=True)
    accs, ious, dices, rows = [], [], [], []

    for case_id in test_case_ids:
        pred_path = pred_dir / (case_id + ".png")
        if not pred_path.exists():
            for ext in [".nii.gz", ".mha", ".nrrd"]:
                p = pred_dir / (case_id + ext)
                if p.exists():
                    pred_path = p
                    break
        if not pred_path.exists():
            print("    SKIP (no pred): " + case_id)
            continue

        gt_path = find_file(case_id, masks_dir)
        if gt_path is None:
            print("    SKIP (no GT): " + case_id)
            continue

        patch_path = find_file(case_id + "_0000", test_images_dir) or \
                     find_file(case_id, test_images_dir)

        pred_arr  = load_pred(pred_path)
        gt_arr    = load_gt(gt_path)
        patch_arr = load_patch(patch_path) if patch_path else \
                    np.zeros((*pred_arr.shape, 3), dtype=np.uint8)

        if patch_arr.shape[:2] != pred_arr.shape:
            patch_arr = np.array(Image.fromarray(patch_arr).resize(
                (pred_arr.shape[1], pred_arr.shape[0]), Image.BILINEAR))

        acc, iou, dice = compute_metrics(pred_arr, gt_arr)
        accs.append(acc); ious.append(iou); dices.append(dice)
        rows.append({"case": case_id,
                     "acc":  round(acc,  4),
                     "iou":  round(iou,  4),
                     "dice": round(dice, 4)})

        save_viz(patch_arr, gt_arr, pred_arr, iou, dice, acc,
                 viz_dir / (case_id + "_viz.png"), case_id)

    mean_acc  = float(np.mean(accs))  if accs  else None
    mean_iou  = float(np.mean(ious))  if ious  else None
    mean_dice = float(np.mean(dices)) if dices else None
    return {
        "test_loss":  round(1 - mean_dice, 4) if mean_dice else None,
        "test_acc":   round(mean_acc,  4)      if mean_acc  else None,
        "test_iou":   round(mean_iou,  4)      if mean_iou  else None,
        "test_dice":  round(mean_dice, 4)      if mean_dice else None,
    }, rows


# ── Summary ────────────────────────────────────────────────────────────────────

def _write_summary(fold_results, seed, path):
    """
    Write summary.csv. fold column uses the fold_label from each result
    (e.g. fold0, fold1, fold2 ... or fold1, fold2 if --fold not specified).
    """
    rows = []
    for r in fold_results:
        row = {"fold": r["fold_label"], "seed": seed}
        for col in SUMMARY_COLS[2:]:
            val = r.get(col)
            row[col] = round(val, 4) if isinstance(val, float) else "-"
        rows.append(row)

    # mean / std only meaningful when multiple folds
    mean_row = {"fold": "mean", "seed": "-"}
    std_row  = {"fold": "std",  "seed": "-"}
    for col in SUMMARY_COLS[2:]:
        vals = [r[col] for r in rows if isinstance(r[col], float)]
        mean_row[col] = round(float(np.mean(vals)), 4) if vals else "-"
        std_row[col]  = round(float(np.std(vals)),  4) if vals else "-"

    write_header = not Path(path).exists()   # only write header if file is new
    with open(path, "a", newline="") as f:   # "a" = append, not overwrite
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print("\nSummary: " + str(path))
    header = "fold   seed   " + "  ".join(c.ljust(12) for c in SUMMARY_COLS[2:])
    print(header)
    print("-" * len(header))
    display = rows + ([mean_row, std_row] if len(rows) > 1 else [])
    for row in display:
        print(str(row["fold"]).ljust(7) +
              str(row["seed"]).ljust(7) +
              "  ".join(str(row[c]).ljust(12) for c in SUMMARY_COLS[2:]))


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_id",   type=int, default=101)
    p.add_argument("--dataset_name", default="WSIPatches")
    p.add_argument("--config",       default="2d")
    p.add_argument("--masks_dir",    required=True)
    p.add_argument("--splits_dir",   required=True,
                   help="Single CSV file OR folder of CSV files")
    p.add_argument("--output_dir",   default="Results")
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--checkpoint",   default=None,
                   help="'best', 'final', or full path to .pth file")
    p.add_argument("--fold",         default=None,
                   help="Fold label to use for output dir and summary column. "
                        "e.g. fold0, fold1, fold2. "
                        "If omitted, uses fold1/fold2/... based on CSV order.")
    return p.parse_args()


def main():
    args = parse_args()

    raw_root     = Path(os.environ["nnUNet_raw"])
    results_root = Path(os.environ["nnUNet_results"])
    masks_dir    = Path(args.masks_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_tag             = "Dataset" + str(args.dataset_id).zfill(3) + "_" + args.dataset_name
    dataset_folder     = raw_root / ds_tag
    test_images_dir    = dataset_folder / "imagesTs"
    test_splits_path   = dataset_folder / "test_splits.json"
    nnunet_trainer_dir = results_root / ds_tag / ("nnUNetTrainer__nnUNetPlans__" + args.config)

    with open(test_splits_path) as f:
        test_splits = json.load(f)

    # ── Collect CSV files ──────────────────────────────────────────────────────
    splits_path = Path(args.splits_dir)
    all_csvs    = [splits_path] if splits_path.is_file() \
                  else sorted(splits_path.glob("*.csv"))
    csv_files   = [f for f in all_csvs if is_valid_split_csv(f)]
    skipped     = [f.name for f in all_csvs if f not in csv_files]
    print("Using " + str(len(csv_files)) + " CSV file(s)")
    if skipped:
        print("Skipped (no train/val/test cols): " + str(skipped))

    # Copy CSVs
    csv_out = output_dir / "split_train_val_test"
    csv_out.mkdir(exist_ok=True)
    for f in csv_files:
        shutil.copy2(f, csv_out / f.name)

    # ── Checkpoint flag ────────────────────────────────────────────────────────
    ckpt_flag = None
    if args.checkpoint:
        if args.checkpoint in ("best",  "checkpoint_best.pth"):
            ckpt_flag = "checkpoint_best.pth"
        elif args.checkpoint in ("final", "checkpoint_final.pth"):
            ckpt_flag = "checkpoint_final.pth"
        else:
            ckpt_flag = args.checkpoint
    print("Checkpoint : " + (str(ckpt_flag) if ckpt_flag else "default (checkpoint_final.pth)"))

    # ── Resolve fold label and nnUNet fold index ───────────────────────────────
    # --fold fold0 → fold_label="fold0", fold_idx=0, test key="fold1" (1-based in json)
    # --fold fold1 → fold_label="fold1", fold_idx=1, test key="fold2"
    # If --fold not given, iterate all CSVs as fold1, fold2, ...
    if args.fold is not None:
        fold_label = args.fold.lower()                       # e.g. "fold1"
        fold_idx   = int(re.sub(r"[^0-9]", "", fold_label)) # e.g. 1
        # test_splits keys are "fold1","fold2"... (1-based from prepare script)
        # when user passes fold0 → nnUNet fold 0 → test key "fold1"
        # when user passes fold1 → nnUNet fold 1 → test key "fold2"
        test_key   = fold_label
        csv_iter   = [(fold_label, fold_idx, test_key, csv_files[0]
                       if len(csv_files) == 1 else csv_files[fold_idx])]
    else:
        csv_iter = [
            ("fold" + str(i + 1), i, "fold" + str(i + 1), csv_path)
            for i, csv_path in enumerate(csv_files)
        ]

    fold_results = []

    for fold_label, fold_idx, test_key, csv_path in csv_iter:
        test_case_ids = test_splits.get(test_key, [])

        print("\n" + "="*60)
        print("Fold label : " + fold_label +
              "  (nnUNet fold_" + str(fold_idx) + ")" +
              "  | CSV: " + csv_path.name +
              "  | test cases: " + str(len(test_case_ids)))
        print("="*60)

        nnunet_fold_dir = nnunet_trainer_dir / ("fold_" + str(fold_idx))

        # Output under output_dir/<fold_label>/
        fold_out_dir = output_dir / fold_label
        ckpt_out_dir = fold_out_dir / "checkpoints"
        pred_out_dir = fold_out_dir / "predictions"
        viz_out_dir  = fold_out_dir / "visualizations"
        fold_out_dir.mkdir(exist_ok=True)
        ckpt_out_dir.mkdir(exist_ok=True)

        # Copy checkpoints
        for src_name, dst_name in [("checkpoint_best.pth",  "best_model.pth"),
                                    ("checkpoint_final.pth", "last_model.pth")]:
            src = nnunet_fold_dir / src_name
            if src.exists():
                shutil.copy2(src, ckpt_out_dir / dst_name)
                print("  Copied: " + src_name + " -> " + dst_name)
            else:
                print("  WARNING: " + src_name + " not found at " + str(src))

        train_metrics = parse_nnunet_training_log(nnunet_fold_dir)
        val_metrics   = parse_nnunet_val_summary(nnunet_fold_dir)

        # Resolve checkpoint path
        resolved_ckpt = None
        if ckpt_flag:
            resolved_ckpt = ckpt_flag \
                if (os.path.isabs(ckpt_flag) or ckpt_flag.startswith(".")) \
                else str(nnunet_fold_dir / ckpt_flag)

        print("  Running inference (nnUNet fold_" + str(fold_idx) + ")...")
        run_test_inference(args.dataset_id, args.config, fold_idx,
                           test_images_dir, pred_out_dir,
                           args.device, checkpoint_path=resolved_ckpt)

        print("  Evaluating and creating visualizations...")
        test_metrics, per_case = evaluate_fold(
            pred_out_dir, test_images_dir, masks_dir,
            test_case_ids, viz_out_dir)

        print("  Saved " + str(len(per_case)) + " visualizations -> " + str(viz_out_dir))

        # Per-fold metrics.csv
        with open(fold_out_dir / "metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case", "acc", "iou", "dice"])
            writer.writeheader()
            writer.writerows(per_case)
            writer.writerow({"case": "mean",
                             "acc":  test_metrics.get("test_acc"),
                             "iou":  test_metrics.get("test_iou"),
                             "dice": test_metrics.get("test_dice")})

        fold_results.append({
            "fold_label": fold_label,   # used in summary fold column
            **train_metrics,
            **val_metrics,
            **test_metrics,
        })
        print("  test_iou=" + str(test_metrics.get("test_iou")) +
              "  test_dice=" + str(test_metrics.get("test_dice")))

    _write_summary(fold_results, args.seed, output_dir / "summary.csv")
    print("\nDone. Results in: " + str(output_dir))


if __name__ == "__main__":
    main()