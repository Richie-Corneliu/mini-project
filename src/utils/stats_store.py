"""Daily vehicle-count persistence for the ATCS workers.

One snapshot per WIB day (Asia/Jakarta, UTC+7) in logs/daily_stats.json. At
midnight the finished day is archived to logs/archive/daily_stats_<date>.json
and the active state restarts at zero, so a restart mid-day keeps counting
from the restored baseline instead of dropping back to 0.

Writes go through <file>.tmp + os.replace: the process can die mid-write and
the next startup still reads a complete snapshot. One lock guards the state
dict and the file, because AI worker threads record counts while FastAPI
serializes the same dict from another thread.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .logger import get_logger

log = get_logger(__name__)

WIB = timezone(timedelta(hours=7))
DEFAULT_STATS_FILE = "logs/daily_stats.json"
DEFAULT_SAVE_INTERVAL_SEC = 5.0

# ZoneCounter bucket label -> stored/API key. The counter calls it "truk/bus"
# while the snapshot schema (main.py, server.py) calls it "truk".
LABEL_KEYS = (("motor", "motor"), ("mobil", "mobil"), ("truk/bus", "truk"))


def today_wib() -> str:
    return datetime.now(WIB).strftime("%Y-%m-%d")


def hour_wib() -> str:
    return datetime.now(WIB).strftime("%H")


def blank_node() -> dict:
    return {"total": 0, "counts": {"motor": 0, "mobil": 0, "truk": 0},
            "hourly_counts": {}}


def normalize_node(raw) -> dict:
    """Any JSON blob -> the node schema; unknown/malformed values mean 0."""
    node = blank_node()
    if not isinstance(raw, dict):
        return node
    counts = raw.get("counts") if isinstance(raw.get("counts"), dict) else {}
    for _, key in LABEL_KEYS:
        try:
            node["counts"][key] = max(0, int(counts.get(key, 0)))
        except (TypeError, ValueError):
            pass
    try:
        node["total"] = max(0, int(raw.get("total", sum(node["counts"].values()))))
    except (TypeError, ValueError):
        node["total"] = sum(node["counts"].values())
    hourly = raw.get("hourly_counts")
    if isinstance(hourly, dict):
        for hour, bucket in hourly.items():
            if not isinstance(bucket, dict):
                continue
            clean = {}
            for _, key in LABEL_KEYS:
                try:
                    clean[key] = max(0, int(bucket.get(key, 0)))
                except (TypeError, ValueError):
                    clean[key] = 0
            node["hourly_counts"][str(hour)] = clean
    return node


def counter_counts(counter) -> dict:
    """Live ZoneCounter per-class totals in the stored schema."""
    totals = counter.total_by_class
    return {key: int(totals.get(label, 0)) for label, key in LABEL_KEYS}


def seed_counter(counter, node) -> None:
    """Restore a day's cumulative counts onto a fresh ZoneCounter.

    Only the numbers move: tracker ids, per-zone latches and the dedup log
    start empty, so the first frame after a restart counts normally. A later
    `_reattribute` shifts these same class buckets and leaves `total` alone,
    which is why the restored split has to be exact rather than just a total.
    """
    node = normalize_node(node)
    for label, key in LABEL_KEYS:
        n = node["counts"][key]
        if n:
            counter.total_by_class[label] = n
    counter.total = max(node["total"], sum(node["counts"].values()))


def fold_hourly_counts(hourly_counts: dict, events, hour=None) -> None:
    """Fold count events into the WIB hour buckets, in place.

    A re-attribution carries `reattributed_from`: the delta moves out of the
    old class bucket instead of adding a second vehicle, so an hour whose
    pickup flipped truck -> car does not report one vehicle too many.
    """
    if not events:
        return
    hour = hour or hour_wib()
    bucket = hourly_counts.setdefault(hour, {"motor": 0, "mobil": 0, "truk": 0})
    for ev in events:
        name = ev["class_name"]
        key = "truk" if name.startswith("truk") else name
        old = ev.get("reattributed_from")
        if old:
            okey = "truk" if old.startswith("truk") else old
            if okey in bucket:
                bucket[okey] = max(0, bucket[okey] - 1)
        if key in bucket:   # unknown label: skip, never crash counting
            bucket[key] += 1


class StatsStore:
    """Thread-safe daily snapshot: in-memory state + atomic JSON on disk."""

    def __init__(self, path=None, archive_dir=None,
                 save_interval_sec=DEFAULT_SAVE_INTERVAL_SEC):
        self.path = Path(path or DEFAULT_STATS_FILE)
        self.archive_dir = (Path(archive_dir) if archive_dir
                            else self.path.parent / "archive")
        self.save_interval = max(0.0, float(save_interval_sec))
        self._lock = threading.RLock()
        self._last_save = 0.0
        self._dirty = False
        self.state = self._empty()

    @classmethod
    def from_config(cls, cfg) -> "StatsStore":
        cfg = cfg or {}
        return cls(cfg.get("stats_file") or DEFAULT_STATS_FILE,
                   cfg.get("archive_dir"),
                   cfg.get("save_interval_sec", DEFAULT_SAVE_INTERVAL_SEC))

    @staticmethod
    def _empty(date=None) -> dict:
        return {"date": date or today_wib(),
                "updated_at": datetime.now(WIB).isoformat(timespec="seconds"),
                "nodes": {}}

    # ---------------------------------------------------------------- startup

    def load(self) -> dict:
        """Restore today's snapshot, archiving a stale one. Never raises."""
        data = None
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as e:   # JSONDecodeError is a ValueError
            log.warning("daily stats %s unreadable (%s); starting from zero",
                        self.path, e)
        with self._lock:
            self._dirty = False
            if not isinstance(data, dict) or not data.get("date"):
                self.state = self._empty()
                return self.snapshot()
            date = str(data["date"])
            nodes = data.get("nodes") if isinstance(data.get("nodes"), dict) else {}
            if date == today_wib():
                self.state = {
                    "date": date,
                    "updated_at": str(data.get("updated_at")
                                      or datetime.now(WIB).isoformat(timespec="seconds")),
                    "nodes": {str(nid): normalize_node(n) for nid, n in nodes.items()},
                }
                log.info("daily stats restored for %s: %d node(s)",
                         date, len(self.state["nodes"]))
                return self.snapshot()
            # A finished day: keep its snapshot, start today at zero.
            self._archive_document(data, date)
            self.state = self._empty()
            log.info("daily stats for %s archived; %s starts at zero",
                     date, self.state["date"])
            return self.snapshot()

    # ------------------------------------------------------------------ state

    def snapshot(self) -> dict:
        with self._lock:
            return {"date": self.state["date"],
                    "updated_at": self.state["updated_at"],
                    "nodes": {nid: {"total": n["total"],
                                    "counts": dict(n["counts"]),
                                    "hourly_counts": {h: dict(b) for h, b in
                                                      n["hourly_counts"].items()}}
                              for nid, n in self.state["nodes"].items()}}

    def node(self, node_id) -> dict:
        with self._lock:
            return normalize_node(self.state["nodes"].get(node_id))

    def record(self, node_id, counter, hourly_counts) -> bool:
        """Copy the live counter into the in-memory snapshot (no disk I/O)."""
        counts = counter_counts(counter)
        hourly = {str(h): {k: int(b.get(k, 0)) for _, k in LABEL_KEYS}
                  for h, b in hourly_counts.items() if isinstance(b, dict)}
        total = int(counter.total)
        with self._lock:
            node = self.state["nodes"].setdefault(node_id, blank_node())
            if (node["total"], node["counts"], node["hourly_counts"]) \
                    == (total, counts, hourly):
                return False
            node["total"] = total
            node["counts"] = counts
            node["hourly_counts"] = hourly
            self._dirty = True
            return True

    # ------------------------------------------------------------------- disk

    def archive_path(self, date: str) -> Path:
        return self.archive_dir / f"daily_stats_{date}.json"

    def maybe_save(self, force: bool = False) -> bool:
        """Write at most once per save_interval; `force` ignores the throttle."""
        with self._lock:
            if not self._dirty:
                return False
            if not force and time.monotonic() - self._last_save < self.save_interval:
                return False
            self._write()
            return True

    def save(self) -> None:
        with self._lock:
            self._write()

    def _write(self) -> None:
        self.state["updated_at"] = datetime.now(WIB).isoformat(timespec="seconds")
        self._write_document(self.state, self.path)
        self._last_save = time.monotonic()
        self._dirty = False

    @staticmethod
    def _write_document(doc, path) -> None:
        """tmp + fsync + os.replace: never a half-written snapshot."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _archive_document(self, doc, date: str) -> Path:
        path = self.archive_path(date)
        try:
            self._write_document(doc, path)
        except OSError as e:
            log.warning("cannot archive daily stats to %s (%s)", path, e)
        return path

    # --------------------------------------------------------------- rollover

    def rollover(self):
        """Close the active WIB day and open today at zero.

        Returns {"from", "to", "archive"} on a day change, else None. The
        caller is responsible for zeroing the live counter and hourly buckets.
        """
        with self._lock:
            prev = self.state.get("date")
            today = today_wib()
            if not prev or prev == today:
                return None
            path = self._archive_document(self.snapshot(), prev)
            self.state = self._empty(today)
            self._write()
            return {"from": prev, "to": today, "archive": path}
