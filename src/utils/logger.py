"""Centralized logging + per-count CSV audit trail for ground-truth checks."""
from __future__ import annotations

import csv
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_CONFIGURED = False


def setup_logging(level: str = "INFO", log_file: Optional[str] = None,
                  file_max_bytes: int = 5 * 1024 * 1024,
                  file_backup_count: int = 3) -> None:
    """Idempotent root-logger setup. Safe to call from any entry point."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        root.setLevel(level.upper())
        return

    root.setLevel(level.upper())
    fmt = logging.Formatter(_DEFAULT_FMT)

    stream_h = logging.StreamHandler(sys.stdout)
    stream_h.setFormatter(fmt)
    root.addHandler(stream_h)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_h = RotatingFileHandler(
            path, maxBytes=file_max_bytes, backupCount=file_backup_count, encoding="utf-8"
        )
        file_h.setFormatter(fmt)
        root.addHandler(file_h)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class AuditLogger:
    """One row per accepted count event -> logs/count_audit.csv."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        if self._f.tell() == 0:
            self._w.writerow(["timestamp", "track_id", "class_name",
                              "confidence", "centroid_x", "centroid_y"])

    def write(self, event: dict) -> None:
        self._w.writerow([f"{time.time():.3f}", event["track_id"],
                          event["class_name"], f"{event['confidence']:.3f}",
                          f"{event['cx']:.1f}", f"{event['cy']:.1f}"])
        self._f.flush()

    def close(self) -> None:
        self._f.close()
