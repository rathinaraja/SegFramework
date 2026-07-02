"""
utils/train_utils.py
---------------------
Training process: loss functions, optimizer, scheduler, and the
ModelProcess class that handles train/eval/test loop, checkpointing,
and logging for ALL segmentation models.

Since every model (UNet, SegNet, nnU-Net, AttentionUNet, UNet++)
has the same training loop, a single class handles them all.
The model architecture lives in modules/ — this file only deals
with how to train it.

If a future model needs custom training logic, subclass ModelProcess
and override train_one_epoch() or evaluate().
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from utils.metrics import MetricTracker, pixel_accuracy, mean_iou, dice_score
from utils.logger  import get_logger, CSVLogger

# ── Loss functions ─────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, n_classes: int, eps: float = 1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.eps       = eps
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        loss  = 0.0
        for cls in range(self.n_classes):
            p     = probs[:, cls]
            t     = (targets == cls).float()
            loss += 1 - (2 * (p * t).sum() + self.eps) / (p.sum() + t.sum() + self.eps)
        return loss / self.n_classes

def build_loss(name: str, n_classes: int) -> nn.Module:
    name = name.lower()
    if name == "cross_entropy": return nn.CrossEntropyLoss()
    if name == "dice":          return DiceLoss(n_classes)
    if name == "dice_ce":
        ce, dice = nn.CrossEntropyLoss(), DiceLoss(n_classes)
        return lambda logits, targets: ce(logits, targets) + dice(logits, targets)
    raise ValueError(f"Unknown loss '{name}'. Options: cross_entropy | dice | dice_ce")

def build_optimizer(cfg: dict, model: nn.Module) -> optim.Optimizer:
    name = cfg["optimizer"]["name"].lower()
    lr   = cfg["training"]["learning_rate"]
    wd   = cfg["training"].get("weight_decay", 0)
    if name == "adam":  return optim.Adam(model.parameters(),  lr=lr, weight_decay=wd)
    if name == "adamw": return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":   return optim.SGD(
        model.parameters(), lr=lr, weight_decay=wd,
        momentum=cfg["optimizer"].get("momentum", 0.9))
    raise ValueError(f"Unknown optimizer '{name}'. Options: adam | adamw | sgd")

def build_scheduler(cfg: dict, optimizer: optim.Optimizer):
    s    = cfg["scheduler"]
    name = s["name"].lower()
    if name == "cosine":  return optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"], eta_min=s.get("min_lr", 1e-6))
    if name == "step":    return optim.lr_scheduler.StepLR(
        optimizer, step_size=s.get("step_size", 30), gamma=s.get("gamma", 0.1))
    if name == "plateau": return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=s.get("gamma", 0.1))
    raise ValueError(f"Unknown scheduler '{name}'. Options: cosine | step | plateau")

# ── Model process ──────────────────────────────────────────────────────────────
class ModelProcess:
    """
    Unified training / evaluation / inference handler for all models.
    Usage (from train.py):
        from utils.train_utils import ModelProcess
        process = ModelProcess(model, cfg, device, checkpoint_dir=ckpt_dir, log_dir=fold_dir)
        process.train(train_loader, val_loader=val_loader, test_loader=test_loader)
    To add a model with custom training logic:
        class MyModelProcess(ModelProcess):
            def train_one_epoch(self, loader, epoch):
                ...  # custom logic
    """
    def __init__(self, model: nn.Module, cfg: dict, device: torch.device,
                 checkpoint_dir: str, log_dir: str):
        self.model     = model.to(device)
        self.cfg       = cfg
        self.device    = device
        self.n_classes = cfg["model"]["n_classes"]
        self.amp       = cfg["training"].get("amp", True)
        self.criterion = build_loss(cfg["training"].get("loss", "cross_entropy"), self.n_classes)
        self.optimizer = build_optimizer(cfg, model)
        self.scheduler = build_scheduler(cfg, self.optimizer)
        self.scaler    = GradScaler('cuda', enabled=self.amp)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir,        exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        model_name          = cfg["model"]["name"]
        self.logger         = get_logger(f"{model_name}_{os.path.basename(log_dir)}", log_dir)
        self.csv_logger     = CSVLogger(os.path.join(log_dir, "metrics.csv"))
        self.best_loss    = float("inf")
        self.patience_ctr = 0
        self.patience     = cfg["training"].get("early_stopping_patience", 15)
        
    # ── One training epoch ────────────────────────────────────────────────────
    def train_one_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        tracker = MetricTracker()

        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device,  non_blocking=True)

            self.optimizer.zero_grad()
            with autocast('cuda', enabled=self.amp):
                logits = self.model(images)
                loss   = self.criterion(logits, masks)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            preds = logits.argmax(dim=1).detach().cpu()
            n     = images.size(0)
            tracker.update("train_loss", loss.item(),                                    n=n)
            tracker.update("train_acc",  pixel_accuracy(preds, masks.cpu()),             n=n)
            tracker.update("train_iou",  mean_iou(preds, masks.cpu(), self.n_classes),   n=n)
            tracker.update("train_dice", dice_score(preds, masks.cpu(), self.n_classes), n=n)
        return tracker.summary()

    # ── Evaluation ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prefix: str = "val") -> dict:
        """prefix = 'val' or 'test' — used as metric key prefix in CSV."""
        if loader is None or len(loader) == 0:
            return {}

        self.model.eval()
        tracker = MetricTracker()
        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks  = masks.to(self.device,  non_blocking=True)
            with autocast('cuda', enabled=self.amp):
                logits = self.model(images)
                loss   = self.criterion(logits, masks)
            preds = logits.argmax(dim=1).cpu()
            m_cpu = masks.cpu()
            n     = images.size(0)
            tracker.update(f"{prefix}_loss", loss.item(),                              n=n)
            tracker.update(f"{prefix}_acc",  pixel_accuracy(preds, m_cpu),             n=n)
            tracker.update(f"{prefix}_iou",  mean_iou(preds, m_cpu, self.n_classes),   n=n)
            tracker.update(f"{prefix}_dice", dice_score(preds, m_cpu, self.n_classes), n=n)
        return tracker.summary()

    # ── Full training loop ────────────────────────────────────────────────────
    def train(self, train_loader: DataLoader, val_loader=None, test_loader=None):
        """
        Metrics written to metrics.csv depend on eval_mode:
          train_val_test  → train_* + val_* every epoch;
                            test_* written once (final row) after loop with best model.
          train_val       → train_* + val_* every epoch;
                            final row re-runs val with best model, labelled 'final_val_as_test'.
          train_test      → train_* + test_* every epoch;
                            final row re-runs test with best model, labelled 'final_test'.
          training_only   → train_* every epoch; no eval rows at all.
        """
        epochs    = self.cfg["training"]["epochs"]
        eval_mode = self.cfg["training"].get("eval_mode", "train_val_test").lower()
        has_val   = val_loader  is not None and len(val_loader)  > 0
        has_test  = test_loader is not None and len(test_loader) > 0

        self.logger.info(
            f"Training {epochs} epochs | eval_mode={eval_mode} | "
            f"val={'yes' if has_val else 'no'} | "
            f"test={'yes' if has_test else 'no'}"
        )
        self.logger.info(
            f"CSV columns per epoch → "
            f"train_*"
            + (" + val_*"  if eval_mode in ("train_val_test", "train_val") else "")
            + (" + test_*" if eval_mode == "train_test"                    else "")
        )

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs} in progress ...")
            
            # ── Train ──────────────────────────────────────────────────────
            train_metrics = self.train_one_epoch(train_loader, epoch)

            # ── Val (train_val_test and train_val modes only) ───────────────
            # train_test / training_only: no val computed during the loop.
            if eval_mode in ("train_val_test", "train_val") and has_val:
                val_metrics = self.evaluate(val_loader, prefix="val")
            else:
                val_metrics = {}

            # ── Test per-epoch (train_test mode only) ──────────────────────
            # train_val_test: holdout test deferred to post-loop best model.
            # train_val     : val doubles as test, reported post-loop only.
            # training_only : no test at all.
            if eval_mode == "train_test" and has_test:
                test_metrics = self.evaluate(test_loader, prefix="test")
            else:
                test_metrics = {}

            # ── Scheduler ──────────────────────────────────────────────────
            sched_metric = (val_metrics.get("val_loss") or test_metrics.get("test_loss") or train_metrics["train_loss"])
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(sched_metric)
            else:
                self.scheduler.step()

            # ── CSV — columns gated by eval_mode ───────────────────────────
            # Row always has: epoch, lr, train_loss, train_acc, train_iou, train_dice
            # + val_*   when eval_mode in (train_val_test, train_val)
            # + test_*  when eval_mode == train_test
            row = {"epoch": epoch, "lr":    self.optimizer.param_groups[0]["lr"], **train_metrics, **val_metrics, **test_metrics,}
            self.csv_logger.log(row)

            # ── Console ────────────────────────────────────────────────────
            msg = (f"Epoch {epoch:03d}/{epochs} | "
                   f"train_loss={train_metrics['train_loss']:.4f} "
                   f"train_iou={train_metrics['train_iou']:.4f} "
                   f"train_dice={train_metrics['train_dice']:.4f}")
            if val_metrics:
                msg += (f" | val_loss={val_metrics['val_loss']:.4f} "
                        f"val_iou={val_metrics['val_iou']:.4f} "
                        f"val_dice={val_metrics['val_dice']:.4f}")
            if test_metrics:
                msg += (f" | test_loss={test_metrics['test_loss']:.4f} "
                        f"test_iou={test_metrics['test_iou']:.4f} "
                        f"test_dice={test_metrics['test_dice']:.4f}")
            self.logger.info(msg)

            # ── Checkpointing ───────────────────────────────────────────────
            # Save metrics dict contains only the keys relevant to eval_mode
            # so train.py / summary.csv only sees columns it should.
            ckpt_metrics = {**train_metrics, **val_metrics, **test_metrics}
            monitor = (val_metrics.get("val_loss")
                       or test_metrics.get("test_loss")
                       or train_metrics["train_loss"])
            self._save_checkpoints(epoch, monitor, ckpt_metrics)

            # ── Early stopping ──────────────────────────────────────────────
            if self.patience_ctr >= self.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        # ── Post-training final evaluation with best model ─────────────────
        #
        # train_val_test → load best model, run holdout test once, append row
        # train_val      → load best model, re-run val as test, append row
        # train_test     → load best model, re-run test once more, append row
        # training_only  → nothing
        #
        # All final rows go into metrics.csv (same file) so there is one
        # unified file per fold. The 'epoch' column value distinguishes them.

        if eval_mode == "train_val_test" and has_test:
            self.logger.info("Running final holdout test with best model …")
            self.load_checkpoint(os.path.join(self.checkpoint_dir, "best_model.pth"))
            test_metrics = self.evaluate(test_loader, prefix="test")
            self._log_final_metrics(eval_mode, train_metrics={}, val_metrics={}, test_metrics=test_metrics)
            # Append final test row — train/val columns intentionally absent
            self.csv_logger.log({"epoch": "final_test", **test_metrics})

        elif eval_mode == "train_val" and has_val:
            self.logger.info("Running final eval on val set (val=test) with best model …")
            self.load_checkpoint(os.path.join(self.checkpoint_dir, "best_model.pth"))
            # Evaluate with prefix="test" so columns read test_* in the final row
            final_metrics = self.evaluate(val_loader, prefix="test")
            self._log_final_metrics(eval_mode, train_metrics={}, val_metrics={}, test_metrics=final_metrics)
            self.csv_logger.log({"epoch": "final_val_as_test", **final_metrics})

        elif eval_mode == "train_test" and has_test:
            self.logger.info("Running final test with best model …")
            self.load_checkpoint(os.path.join(self.checkpoint_dir, "best_model.pth"))
            test_metrics = self.evaluate(test_loader, prefix="test")
            self._log_final_metrics(eval_mode, train_metrics={}, val_metrics={}, test_metrics=test_metrics)
            self.csv_logger.log({"epoch": "final_test", **test_metrics})

        elif eval_mode == "training_only":
            self.logger.info("training_only mode — skipping post-training evaluation.")

        self.logger.info("Training complete.")

    def _log_final_metrics(self, eval_mode: str, train_metrics: dict, val_metrics:   dict, test_metrics:  dict):
        """
        Log a clean summary line for the final evaluation.

        What is printed depends on eval_mode:
          train_val_test  → test_* only  (holdout set, run once)
          train_val       → test_* only  (val set re-labelled as test)
          train_test      → test_* only  (test set, best-model re-run)
          training_only   → (never called)
        """
        parts = []

        if train_metrics:
            parts.append(
                f"train | loss={train_metrics['train_loss']:.4f} "
                f"acc={train_metrics['train_acc']:.4f} "
                f"iou={train_metrics['train_iou']:.4f} "
                f"dice={train_metrics['train_dice']:.4f}"
            )

        if val_metrics:
            parts.append(
                f"val   | loss={val_metrics['val_loss']:.4f} "
                f"acc={val_metrics['val_acc']:.4f} "
                f"iou={val_metrics['val_iou']:.4f} "
                f"dice={val_metrics['val_dice']:.4f}"
            )

        if test_metrics:
            label = "val→test" if eval_mode == "train_val" else "test"
            parts.append(
                f"{label} | loss={test_metrics['test_loss']:.4f} "
                f"acc={test_metrics['test_acc']:.4f} "
                f"iou={test_metrics['test_iou']:.4f} "
                f"dice={test_metrics['test_dice']:.4f}"
            )

        header = f"── Final Results [{eval_mode}] "
        self.logger.info(header + "─" * max(0, 60 - len(header)))
        for p in parts:
            self.logger.info(f"  {p}")
        self.logger.info("─" * 60)

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with autocast('cuda', enabled=self.amp):
            logits = self.model(images.to(self.device))
        return logits.argmax(dim=1).cpu()

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoints(self, epoch: int, monitored_loss: float, metrics: dict):
        """
        Persist last and best checkpoints.
        The metrics dict stored in the checkpoint contains only keys
        relevant to the current eval_mode so train.py/_write_summary()
        sees a clean, mode-specific set of columns:
          training_only   -> train_*
          train_val       -> train_* + val_*
          train_test      -> train_* + test_*
          train_val_test  -> train_* + val_*  (test_* added post-loop by train.py)
        """
        eval_mode = self.cfg["training"].get("eval_mode", "train_val_test").lower()

        def _keep(key: str) -> bool:
            if key.startswith("train_"): return True
            if key.startswith("val_"):   return eval_mode in ("train_val_test", "train_val")
            if key.startswith("test_"):  return eval_mode == "train_test"
            return False

        filtered_metrics = {k: v for k, v in metrics.items() if _keep(k)}

        state = {
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics":         filtered_metrics,
        }
        torch.save(state, os.path.join(self.checkpoint_dir, "last_model.pth"))

        if monitored_loss < self.best_loss:
            self.best_loss    = monitored_loss
            self.patience_ctr = 0
            torch.save(state, os.path.join(self.checkpoint_dir, "best_model.pth"))
            self.logger.info(f"  ✓ Best model saved (loss={monitored_loss:.4f})")
        else:
            self.patience_ctr += 1

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.logger.info(f"Loaded checkpoint: {path}  (epoch {ckpt.get('epoch', '?')})")