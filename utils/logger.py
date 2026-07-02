"""
utils/logger.py
---------------
CSV + console logger. Writes one row per epoch to a CSV for easy plotting.
"""
import csv
import logging
import os
import sys
from datetime import datetime

def get_logger(name: str, log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger  # avoid duplicate handlers
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # File
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

class CSVLogger:
    """
    Appends metric dicts as rows to a CSV file.
    Handles rows with new columns (e.g. the final_test row that adds test_*
    columns not present in per-epoch rows): when new keys are detected the
    file is rewritten with the expanded header so all prior rows get empty
    values for the new columns, then the new row is appended normally.
    """
    def __init__(self, path: str):
        self.path        = path
        self._fieldnames = None

    def log(self, row: dict):
        new_keys     = [k for k in row if k not in (self._fieldnames or [])]
        file_exists  = os.path.exists(self.path)

        # ── First row ever ────────────────────────────────────────────────
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with open(self.path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            return

        # ── Row introduces new columns → rewrite with expanded header ─────
        if new_keys:
            self._fieldnames = self._fieldnames + new_keys
            # Read all existing rows
            existing_rows = []
            if file_exists:
                with open(self.path, newline="") as f:
                    existing_rows = list(csv.DictReader(f))

            # Rewrite file with expanded header; old rows get "" for new cols
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
                writer.writeheader()
                for old_row in existing_rows:
                    writer.writerow({k: old_row.get(k, "") for k in self._fieldnames})
                writer.writerow({k: row.get(k, "") for k in self._fieldnames})
            return

        # ── Normal append ─────────────────────────────────────────────────
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames,
                                    extrasaction="ignore")
            writer.writerow({k: row.get(k, "") for k in self._fieldnames})