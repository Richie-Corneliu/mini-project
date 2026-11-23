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

import copy
import json
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .logger import get_logger

log = get_logger(__name__)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ARCHIVE_GLOB_RE = re.compile(r"^daily_stats_(\d{4}-\d{2}-\d{2})\.json$")
_HHMM_RE = re.compile(r"^\d{2}:\d{2}$")

WIB = timezone(timedelta(hours=7))
DEFAULT_STATS_FILE = "logs/daily_stats.json"
DEFAULT_SAVE_INTERVAL_SEC = 5.0
DEFAULT_MIN_PERSIST_UPTIME_SEC = 900.0

# ZoneCounter bucket label -> stored/API key. The counter calls it "truk/bus"
# while the snapshot schema (main.py, server.py) calls it "truk".
LABEL_KEYS = (("motor", "motor"), ("mobil", "mobil"), ("truk/bus", "truk"))


def today_wib() -> str:
    return datetime.now(WIB).strftime("%Y-%m-%d")


def hour_wib() -> str:
    return datetime.now(WIB).strftime("%H")


def hhmm_wib() -> str:
    return datetime.now(WIB).strftime("%H:%M")


def slot_wib(dt=None) -> str:
    d = dt or datetime.now(WIB)
    return f"{d.hour:02d}:{(d.minute // 15) * 15:02d}"


def all_slot_keys() -> list[str]:
    return [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]


def format_slot_interval(slot_hhmm: str, session_starts=None) -> str:
    """Format 15-minute slot HH:MM as 'HH:MM - HH:EE', annotating any system start."""
    hh, mm = (int(p) for p in slot_hhmm.split(":"))
    start_min = hh * 60 + mm
    end_min = start_min + 15
    end_str = f"{(end_min // 60) % 24:02d}:{end_min % 60:02d}"
    matched = None
    for raw in session_starts or ():
        try:
            sh, sm = (int(p) for p in str(raw).strip().split(":"))
        except ValueError:
            continue
        s_min = sh * 60 + sm
        if start_min <= s_min < end_min and (matched is None or s_min < matched[0]):
            matched = (s_min, f"{sh:02d}:{sm:02d}")
    if matched is not None:
        return f"{matched[1]} (system start) - {end_str}"
    return f"{slot_hhmm} - {end_str}"


def blank_node() -> dict:
    return {"total": 0, "counts": {"motor": 0, "mobil": 0, "truk": 0},
            "hourly_counts": {}, "slot_counts": {}, "session_starts": []}


def _clean_bucket(bucket, with_total: bool = False) -> Optional[dict]:
    if not isinstance(bucket, dict):
        return None
    clean = {}
    for _, key in LABEL_KEYS:
        raw_val = bucket.get(key, bucket.get("truk/bus", 0) if key == "truk" else 0)
        try:
            clean[key] = max(0, int(raw_val))
        except (TypeError, ValueError):
            clean[key] = 0
    if with_total:
        try:
            explicit = max(0, int(bucket.get("total", 0)))
        except (TypeError, ValueError):
            explicit = 0
        clean["total"] = max(explicit, sum(clean.values()))
    return clean


def _normalize_session_starts(raw) -> list[str]:
    out = []
    for item in raw or ():
        s = str(item).strip()
        if _HHMM_RE.match(s) and s not in out:
            out.append(s)
    return out


def normalize_node(raw) -> dict:
    """Any JSON blob -> the node schema; unknown/malformed values mean 0."""
    node = blank_node()
    if not isinstance(raw, dict):
        return node
    counts = raw.get("counts") if isinstance(raw.get("counts"), dict) else {}
    for _, key in LABEL_KEYS:
        try:
            raw_val = counts.get(key, counts.get("truk/bus", 0) if key == "truk" else 0)
            node["counts"][key] = max(0, int(raw_val))
        except (TypeError, ValueError):
            pass
    try:
        node["total"] = max(0, int(raw.get("total", sum(node["counts"].values()))))
    except (TypeError, ValueError):
        node["total"] = sum(node["counts"].values())
    hourly = raw.get("hourly_counts")
    if isinstance(hourly, dict):
        for hour, bucket in hourly.items():
            clean = _clean_bucket(bucket, with_total=False)
            if clean is not None:
                node["hourly_counts"][str(hour)] = clean
    slots = raw.get("slot_counts")
    if isinstance(slots, dict):
        for slot, bucket in slots.items():
            clean = _clean_bucket(bucket, with_total=True)
            if clean is not None:
                node["slot_counts"][str(slot)] = clean
    node["session_starts"] = _normalize_session_starts(raw.get("session_starts"))
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


def _apply_event_bucket(bucket: dict, ev: dict) -> None:
    name = ev["class_name"]
    key = "truk" if name.startswith("truk") else name
    old = ev.get("reattributed_from")
    if old:
        okey = "truk" if old.startswith("truk") else old
        if okey in bucket:
            bucket[okey] = max(0, bucket[okey] - 1)
    if key in bucket:   # unknown label: skip, never crash counting
        bucket[key] += 1


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
        _apply_event_bucket(bucket, ev)


def fold_slot_counts(slot_counts: dict, hourly_counts: Optional[dict],
                     events, slot: Optional[str] = None) -> None:
    """Fold count events into 15-min WIB slots and keep hourly_counts in sync."""
    if not events:
        return
    slot_key = slot or slot_wib()
    hh = slot_key[:2]
    sb = slot_counts.setdefault(
        slot_key, {"motor": 0, "mobil": 0, "truk": 0, "total": 0}
    )
    hb = None
    if hourly_counts is not None:
        hb = hourly_counts.setdefault(hh, {"motor": 0, "mobil": 0, "truk": 0})
    for ev in events:
        _apply_event_bucket(sb, ev)
        if hb is not None:
            _apply_event_bucket(hb, ev)
    sb["total"] = sb["motor"] + sb["mobil"] + sb["truk"]


class StatsStore:
    """Thread-safe daily snapshot: in-memory state + atomic JSON on disk."""

    def __init__(self, path=None, archive_dir=None,
                 save_interval_sec=DEFAULT_SAVE_INTERVAL_SEC,
                 min_persist_uptime_seconds: float = DEFAULT_MIN_PERSIST_UPTIME_SEC):
        self.path = Path(path or DEFAULT_STATS_FILE)
        self.archive_dir = (Path(archive_dir) if archive_dir
                            else self.path.parent / "archive")
        self.save_interval = max(0.0, float(save_interval_sec))
        self.min_persist_uptime_seconds = max(0.0, float(min_persist_uptime_seconds))
        self._lock = threading.RLock()
        self._started_at = time.monotonic()
        self._last_save = 0.0
        self._dirty = False
        self.state = self._empty()
        self._baseline_state = copy.deepcopy(self.state)
        self._had_baseline_file = self.path.exists()

    @classmethod
    def from_config(cls, cfg) -> "StatsStore":
        cfg = cfg or {}
        if "min_persist_uptime_seconds" in cfg and cfg["min_persist_uptime_seconds"] is not None:
            min_uptime = float(cfg["min_persist_uptime_seconds"])
        elif "min_uptime_minutes" in cfg and cfg["min_uptime_minutes"] is not None:
            min_uptime = float(cfg["min_uptime_minutes"]) * 60.0
        else:
            min_uptime = DEFAULT_MIN_PERSIST_UPTIME_SEC
        return cls(cfg.get("stats_file") or DEFAULT_STATS_FILE,
                   cfg.get("archive_dir"),
                   cfg.get("save_interval_sec", DEFAULT_SAVE_INTERVAL_SEC),
                   min_persist_uptime_seconds=min_uptime)

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
            self._started_at = time.monotonic()
            self._had_baseline_file = self.path.exists()
            if not isinstance(data, dict) or not data.get("date"):
                self.state = self._empty()
                self._baseline_state = copy.deepcopy(self.state)
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
                self._baseline_state = copy.deepcopy(self.state)
                log.info("daily stats restored for %s: %d node(s)",
                         date, len(self.state["nodes"]))
                return self.snapshot()
            # A finished day: keep its snapshot, start today at zero.
            self._archive_document(data, date)
            self.state = self._empty()
            self._baseline_state = copy.deepcopy(self.state)
            self.save(force=True)
            log.info("daily stats for %s archived; %s starts at zero",
                     date, self.state["date"])
            return self.snapshot()

    # ------------------------------------------------------------------ state

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "date": self.state["date"],
                "updated_at": self.state["updated_at"],
                "nodes": {
                    nid: {
                        "total": n["total"],
                        "counts": dict(n["counts"]),
                        "hourly_counts": {h: dict(b) for h, b in n["hourly_counts"].items()},
                        "slot_counts": {s: dict(b) for s, b in n.get("slot_counts", {}).items()},
                        "session_starts": list(n.get("session_starts", [])),
                    }
                    for nid, n in self.state["nodes"].items()
                },
            }

    def get_all(self) -> dict:
        """Alias of snapshot() returning a deep copy of in-memory state."""
        return self.snapshot()

    def session_uptime(self, now: Optional[float] = None) -> float:
        current = time.monotonic() if now is None else float(now)
        return max(0.0, current - self._started_at)

    def _uptime_reached(self, now: Optional[float] = None) -> bool:
        if self.min_persist_uptime_seconds <= 0.0:
            return True
        return self.session_uptime(now) >= self.min_persist_uptime_seconds

    def clear(self, save_to_disk: bool = True) -> dict:
        """Reset today's state to zero and optionally persist immediately."""
        with self._lock:
            self.state = self._empty()
            self._baseline_state = copy.deepcopy(self.state)
            self._dirty = False
            if save_to_disk:
                self._write()
            return self.snapshot()

    def prune_hourly_below(self, min_total: int = 20) -> int:
        """Remove hourly buckets with total vehicles < min_total and recompute node daily totals."""
        threshold = int(min_total)
        pruned = 0
        with self._lock:
            for node in self.state["nodes"].values():
                hourly = node.get("hourly_counts")
                if not isinstance(hourly, dict):
                    continue
                kept = {}
                for hour, bucket in list(hourly.items()):
                    if not isinstance(bucket, dict):
                        continue
                    motor = max(0, int(bucket.get("motor", 0)))
                    mobil = max(0, int(bucket.get("mobil", 0)))
                    truk = max(0, int(bucket.get("truk", bucket.get("truk/bus", 0))))
                    if motor + mobil + truk < threshold:
                        pruned += 1
                    else:
                        kept[str(hour)] = {"motor": motor, "mobil": mobil, "truk": truk}
                node["hourly_counts"] = kept
                slots = node.get("slot_counts")
                if isinstance(slots, dict) and slots:
                    node["slot_counts"] = {
                        str(s): b for s, b in slots.items() if str(s)[:2] in kept
                    }
                recomputed = {
                    key: sum(int(b.get(key, 0)) for b in kept.values())
                    for _, key in LABEL_KEYS
                }
                node["counts"] = recomputed
                node["total"] = sum(recomputed.values())
            if pruned > 0:
                self._dirty = True
            self._baseline_state = copy.deepcopy(self.state)
            return pruned

    def node(self, node_id) -> dict:
        with self._lock:
            return normalize_node(self.state["nodes"].get(node_id))

    def record(self, node_id, counter, hourly_counts,
               slot_counts=None, session_starts=None) -> bool:
        """Copy the live counter into the in-memory snapshot (no disk I/O)."""
        counts = counter_counts(counter)
        hourly = {
            str(h): clean
            for h, b in hourly_counts.items()
            if (clean := _clean_bucket(b, with_total=False)) is not None
        }
        slots = None
        if slot_counts is not None:
            slots = {
                str(s): clean
                for s, b in slot_counts.items()
                if (clean := _clean_bucket(b, with_total=True)) is not None
            }
        starts = None
        if session_starts is not None:
            starts = _normalize_session_starts(session_starts)
        total = int(counter.total)
        with self._lock:
            node = self.state["nodes"].setdefault(node_id, blank_node())
            target_slots = node.get("slot_counts", {}) if slots is None else slots
            target_starts = node.get("session_starts", []) if starts is None else starts
            if (node["total"], node["counts"], node["hourly_counts"],
                    node.get("slot_counts", {}), node.get("session_starts", [])) \
                    == (total, counts, hourly, target_slots, target_starts):
                return False
            node["total"] = total
            node["counts"] = counts
            node["hourly_counts"] = hourly
            node["slot_counts"] = target_slots
            node["session_starts"] = target_starts
            self._dirty = True
            return True

    @staticmethod
    def _doc_has_data(doc) -> bool:
        if not isinstance(doc, dict):
            return False
        for node in (doc.get("nodes") or {}).values():
            norm = normalize_node(node)
            if norm["total"] > 0:
                return True
            if any(sum(b.values()) > 0 for b in norm["hourly_counts"].values()):
                return True
            if any(b.get("total", 0) > 0 for b in norm["slot_counts"].values()):
                return True
        return False

    def recorded_dates(self) -> list[str]:
        """Return sorted YYYY-MM-DD dates with recorded counting data."""
        dates = set()
        with self._lock:
            cur_date = str(self.state.get("date") or "")
            if _DATE_RE.match(cur_date) and self._doc_has_data(self.state):
                dates.add(cur_date)
            archive_dir = self.archive_dir
        if archive_dir.exists():
            for path in archive_dir.glob("daily_stats_*.json"):
                m = _ARCHIVE_GLOB_RE.match(path.name)
                if not m:
                    continue
                d = m.group(1)
                if d in dates:
                    continue
                try:
                    with open(path, encoding="utf-8") as f:
                        doc = json.load(f)
                except (OSError, ValueError):
                    continue
                if self._doc_has_data(doc):
                    dates.add(d)
        return sorted(dates)

    def load_date(self, date_str: str) -> Optional[dict]:
        """Return normalized snapshot for YYYY-MM-DD (today or archive)."""
        if not _DATE_RE.match(str(date_str or "")):
            return None
        with self._lock:
            if date_str == self.state.get("date"):
                return self.snapshot()
            path = self.archive_path(date_str)
        try:
            with open(path, encoding="utf-8") as f:
                raw_doc = json.load(f)
        except (FileNotFoundError, OSError, ValueError):
            return None
        if not isinstance(raw_doc, dict):
            return None
        raw_nodes = raw_doc.get("nodes") if isinstance(raw_doc.get("nodes"), dict) else {}
        return {
            "date": date_str,
            "updated_at": str(raw_doc.get("updated_at") or ""),
            "nodes": {str(nid): normalize_node(n) for nid, n in raw_nodes.items()},
        }

    # ------------------------------------------------------------------- disk

    def archive_path(self, date: str) -> Path:
        return self.archive_dir / f"daily_stats_{date}.json"

    def maybe_save(self, force: bool = False, now: Optional[float] = None) -> bool:
        """Write at most once per save_interval after min_persist_uptime_seconds; `force` ignores both."""
        with self._lock:
            if not self._dirty and not force:
                return False
            if not force:
                if not self._uptime_reached(now):
                    return False
                current = time.monotonic() if now is None else float(now)
                if current - self._last_save < self.save_interval:
                    return False
            self._write(now=now)
            return True

    def save(self, force: bool = False, now: Optional[float] = None) -> bool:
        with self._lock:
            if not force and not self._uptime_reached(now):
                log.info(
                    "skipping daily stats save: session uptime %.1fs < %.1fs",
                    self.session_uptime(now),
                    self.min_persist_uptime_seconds,
                )
                return False
            self._write(now=now)
            return True

    def _write(self, now: Optional[float] = None) -> None:
        self.state["updated_at"] = datetime.now(WIB).isoformat(timespec="seconds")
        self._write_document(self.state, self.path)
        self._last_save = time.monotonic() if now is None else float(now)
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
            self._baseline_state = copy.deepcopy(self.state)
            self.save(force=True)
            return {"from": prev, "to": today, "archive": path}
