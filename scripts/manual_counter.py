"""Live AI-synced manual counter for ground-truth validation.

Manual keys stay the original ones (q/w/l, Shift to decrement, r reset, ESC
quit) and mouse +/- buttons work inside each card. On top of that the window
polls /api/v1/traffic-data once a second in a daemon thread, so the OpenCV
loop never blocks on network I/O, and shows the AI delta plus live accuracy
next to every manual count. SPACE starts/stops a timed benchmark session,
S (or closing the window) writes a JSON + Markdown report through
scripts/evaluate_accuracy.py.

Standalone fallback: when the API is unreachable the HUD reads
"API: OFFLINE (Standalone Mode)" and the counter keeps working.

Run:  python scripts/manual_counter.py [--api-url URL] [--node ID]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

try:
    from scripts.evaluate_accuracy import (compute_accuracy_report,
                                           format_markdown, normalize_counts)
except ImportError:   # launched as `python scripts/manual_counter.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.evaluate_accuracy import (compute_accuracy_report,
                                           format_markdown, normalize_counts)

# Manual-count key -> (BGR accent, canonical evaluate_accuracy key)
CLASSES = (("motor", (0, 220, 90), "motor"),
           ("mobil", (60, 120, 255), "mobil"),
           ("bis/truk", (0, 220, 220), "truk_bus"))
CLASS_NAMES = tuple(name for name, _, _ in CLASSES)
CANON = tuple(canon for _, _, canon in CLASSES)

KEYMAP_INC = {ord("q"): "motor", ord("w"): "mobil", ord("l"): "bis/truk"}
KEYMAP_DEC = {ord("Q"): "motor", ord("W"): "mobil", ord("L"): "bis/truk"}

W, H = 720, 430
FONT = cv2.FONT_HERSHEY_SIMPLEX

INK = (240, 244, 248)
MUTED = (150, 160, 175)
PANEL = (38, 45, 60)
PANEL_HI = (52, 62, 80)
BG = (24, 30, 42)
GREEN = (110, 220, 140)
AMBER = (60, 200, 250)
RED = (90, 90, 240)

POLL_SEC = 1.0
HTTP_TIMEOUT = 1.5

# Filled by render(); consumed by the mouse callback in main().
UI_REGIONS: list = []


def zeros() -> dict:
    return {key: 0 for key in CANON}


def fmt_mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def fetch_traffic_data(api_url: str, timeout: float = HTTP_TIMEOUT) -> dict:
    """GET {api_url}/api/v1/traffic-data -> parsed JSON. Raises on any failure."""
    url = api_url.rstrip("/") + "/api/v1/traffic-data"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class BenchmarkSession:
    """Manual counts + AI snapshots + timed session state, no OpenCV inside."""

    def __init__(self, api_url="http://127.0.0.1:8000", node_id=None,
                 scenario="Live RTSP Validation", clock=time.monotonic):
        self.api_url = api_url
        self.scenario = scenario
        self._clock = clock

        self.counts = {name: 0 for name in CLASS_NAMES}
        self.state = "IDLE"

        self.node_ids: list = []
        self._nodes: dict = {}
        self._requested_node = node_id
        self.node_id = node_id
        self.node_name = node_id or "-"

        self.ai_latest = zeros()
        self.ai_baseline = None
        self.ai_stop_snapshot = None

        self.api_connected = False
        self.api_error = ""

        self.started_at = None
        self.stopped_at = None

        self.toast = ""
        self.toast_until = 0.0
        self._saved_signature = None

        self._stop = threading.Event()
        self._poller = None
        self._lock = threading.Lock()

    # ---- counts ---------------------------------------------------

    def adjust(self, name: str, delta: int = 1) -> int:
        """Clamped at zero; returns the new count."""
        with self._lock:
            self.counts[name] = max(0, self.counts.get(name, 0) + delta)
            return self.counts[name]

    def increment(self, name: str) -> int:
        return self.adjust(name, 1)

    def decrement(self, name: str) -> int:
        return self.adjust(name, -1)

    @property
    def manual_total(self) -> int:
        return sum(self.counts.values())

    def reset(self):
        with self._lock:
            self.counts = {name: 0 for name in CLASS_NAMES}
            self.state = "IDLE"
            self.started_at = self.stopped_at = None
            self.ai_baseline = None
            self.ai_stop_snapshot = None
            self._saved_signature = None
        self.set_toast("Session reset")

    # ---- session --------------------------------------------------

    def start_session(self):
        with self._lock:
            self.state = "RUNNING"
            self.started_at = self._clock()
            self.stopped_at = None
            self.ai_baseline = dict(self.ai_latest)
            self.ai_stop_snapshot = None
        self.set_toast("Benchmark started" if self.api_connected
                       else "Benchmark started (standalone)")

    def stop_session(self):
        with self._lock:
            if self.state != "RUNNING":
                return
            self.state = "STOPPED"
            self.stopped_at = self._clock()
            self.ai_stop_snapshot = dict(self.ai_latest)
        self.set_toast("Benchmark stopped - press S to save")

    def toggle_session(self):
        if self.state == "RUNNING":
            self.stop_session()
        else:
            self.start_session()

    def elapsed_seconds(self, now=None) -> float:
        if self.started_at is None:
            return 0.0
        end = self.stopped_at if self.stopped_at is not None \
            else (now if now is not None else self._clock())
        return max(0.0, end - self.started_at)

    # ---- nodes / API ----------------------------------------------

    def apply_payload(self, payload: dict) -> bool:
        """Consume one /traffic-data response. Returns True on success."""
        nodes = (payload or {}).get("nodes") or {}
        if not isinstance(nodes, dict) or not nodes:
            return False          # nothing to sync against
        with self._lock:
            self._nodes = nodes
            ids = sorted(nodes)
            if ids != self.node_ids:
                self.node_ids = ids
            if self._requested_node and self._requested_node in nodes:
                self.node_id = self._requested_node
            elif self.node_id not in nodes:
                self.node_id = ids[0] if ids else None
            if self.node_id:
                raw = nodes.get(self.node_id) or {}
                self.node_name = raw.get("nama") or self.node_id
                self.ai_latest = normalize_counts(raw)
            self.api_connected = True
            self.api_error = ""
        return True

    def poll_once(self, timeout: float = HTTP_TIMEOUT) -> bool:
        """One blocking fetch; never raises, so the poller thread cannot die."""
        try:
            payload = fetch_traffic_data(self.api_url, timeout=timeout)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
            with self._lock:
                self.api_connected = False
                self.api_error = str(getattr(e, "reason", e))[:80]
            return False
        return self.apply_payload(payload)

    def cycle_node(self) -> str:
        """Next discovered node; only meaningful while IDLE."""
        with self._lock:
            if len(self.node_ids) < 2 or self.state != "IDLE":
                return self.node_id
            i = self.node_ids.index(self.node_id) \
                if self.node_id in self.node_ids else -1
            self.node_id = self.node_ids[(i + 1) % len(self.node_ids)]
            self._requested_node = self.node_id
            raw = self._nodes.get(self.node_id) or {}
            self.node_name = raw.get("nama") or self.node_id
            self.ai_latest = normalize_counts(raw)
            self.ai_baseline = None
            self.ai_stop_snapshot = None
            node_id = self.node_id
        self.set_toast(f"Node -> {node_id}")
        return node_id

    def start_poller(self, interval: float = POLL_SEC):
        if self._poller is not None:
            return
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                self.poll_once()
                self._stop.wait(interval)

        self._poller = threading.Thread(target=loop, daemon=True,
                                        name="atcs-bench-poll")
        self._poller.start()

    def stop_poller(self, join_timeout: float = 2.0):
        self._stop.set()
        if self._poller is not None:
            self._poller.join(join_timeout)
            self._poller = None

    # ---- accuracy ---------------------------------------------------

    def ai_delta(self) -> dict:
        with self._lock:
            if self.state == "IDLE" or self.ai_baseline is None:
                return zeros()
            current = self.ai_stop_snapshot or self.ai_latest
            return {k: max(0, int(current.get(k, 0)) -
                           int(self.ai_baseline.get(k, 0))) for k in CANON}

    def live_report(self) -> dict:
        return compute_accuracy_report(gt=dict(self.counts),
                                       pred=self.ai_delta(),
                                       scenario=self.scenario)

    def _signature(self) -> tuple:
        delta = self.ai_delta()
        return (tuple(self.counts[n] for n in CLASS_NAMES), self.state,
                tuple(delta[k] for k in CANON))

    # ---- persistence --------------------------------------------------

    def save_report(self, output_dir, stamp=None, quiet=False) -> Path:
        """Write benchmark_<node>_<ts>.json + .md; returns the JSON path."""
        report = self.live_report()
        delta = self.ai_delta()
        stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        base = out / f"benchmark_{self.node_id or 'manual'}_{stamp}"

        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "scenario": self.scenario,
            "node_id": self.node_id,
            "node_name": self.node_name,
            "duration_sec": round(self.elapsed_seconds(), 1),
            "api_connected": self.api_connected,
            "manual_counts": dict(self.counts),
            "ai_baseline": self.ai_baseline or zeros(),
            "ai_final": self.ai_stop_snapshot or dict(self.ai_latest),
            "ai_delta": delta,
            "report": report,
        }
        with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        md = "\n".join([
            f"# Benchmark - {self.node_name}",
            "",
            f"- Node: `{self.node_id}`",
            f"- Duration: {fmt_mmss(self.elapsed_seconds())} "
            f"({payload['duration_sec']}s)",
            f"- Timestamp: {payload['timestamp']}",
            f"- API: {'ONLINE' if self.api_connected else 'OFFLINE (Standalone Mode)'}",
            "",
            format_markdown(report),
            "",
        ])
        with open(base.with_suffix(".md"), "w", encoding="utf-8") as f:
            f.write(md)

        self._saved_signature = self._signature()
        if not quiet:
            print(format_markdown(report))
            print(f"\nDisimpan ke: {base}.json / {base}.md")
        self.set_toast(f"Saved to {out}")
        return base.with_suffix(".json")

    def maybe_save_on_exit(self, output_dir) -> Path:
        """Auto-save on quit when something was counted and not yet saved."""
        if self.manual_total == 0 and sum(self.ai_delta().values()) == 0:
            return None
        if self._saved_signature == self._signature():
            return None
        return self.save_report(output_dir, quiet=True)

    # ---- HUD helpers ------------------------------------------------

    def set_toast(self, text: str, seconds: float = 3.0):
        self.toast = text
        self.toast_until = self._clock() + seconds

    def toast_text(self, now=None) -> str:
        now = now if now is not None else self._clock()
        return self.toast if now < self.toast_until else ""

    def status_label(self) -> str:
        if self.state == "RUNNING":
            return f"REC {fmt_mmss(self.elapsed_seconds())}"
        if self.state == "STOPPED":
            return f"STOPPED {fmt_mmss(self.elapsed_seconds())}"
        return "IDLE - Press SPACE"


# ---- Rendering (pure: returns the frame, opens no window) -------------

def _text(img, s, x, y, scale=0.5, color=INK, thick=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, thick,
                cv2.LINE_AA)


def _text_size(s, scale=0.5, thick=1):
    return cv2.getTextSize(s, FONT, scale, thick)[0]


def _pill(img, s, x, y, h=22, fg=GREEN, bg=PANEL, scale=0.45, pad=10):
    w = _text_size(s, scale, 1)[0] + pad * 2
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), fg, 1)
    _text(img, s, x + pad, y + h - 7, scale, fg, 1)
    return x + w


def _button(img, s, x, y, w, h, action, payload=None, fg=INK, active=False):
    bg = PANEL_HI if active else PANEL
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), fg, 1)
    tw = _text_size(s, 0.45, 1)[0]
    _text(img, s, x + (w - tw) // 2, y + h - (h - 12) // 2 - 1, 0.45, fg, 1)
    UI_REGIONS.append(((x, y, x + w, y + h), action, payload))


def render(session: BenchmarkSession, now=None) -> np.ndarray:
    """Draw the whole HUD and return it as a (H, W, 3) uint8 BGR array."""
    UI_REGIONS.clear()
    img = np.full((H, W, 3), BG, np.uint8)
    report = session.live_report()
    delta = session.ai_delta()
    rows = {r["key"]: r for r in report["rows"]}

    # ---- header -------------------------------------------------
    cv2.rectangle(img, (0, 0), (W, 62), PANEL, -1)
    cv2.line(img, (0, 62), (W, 62), (70, 82, 100), 1)
    _text(img, "TRAFFIC MANUAL COUNTER & LIVE BENCHMARK", 16, 26, 0.55, INK, 1)

    node_label = f"Node: {session.node_id or '-'} [N]"
    _text(img, node_label, 16, 50, 0.45, MUTED, 1)
    node_w = _text_size(node_label, 0.45, 1)[0]
    UI_REGIONS.append(((16, 36, 16 + node_w + 8, 56), "cycle", None))

    api_text = "API: ONLINE" if session.api_connected \
        else "API: OFFLINE (Standalone Mode)"
    api_fg = GREEN if session.api_connected else AMBER
    status = session.status_label()
    status_fg = RED if session.state == "RUNNING" else \
        (AMBER if session.state == "STOPPED" else MUTED)
    right_w = _text_size(status, 0.45, 1)[0] + 20
    api_w = _text_size(api_text, 0.45, 1)[0] + 20
    _pill(img, status, W - 14 - right_w, 34, 22, status_fg)
    _pill(img, api_text, W - 14 - right_w - api_w - 8, 34, 22, api_fg)

    # ---- class cards --------------------------------------------
    gap, margin, top = 14, 18, 78
    card_w = (W - 2 * margin - 2 * gap) // 3
    card_h = 196
    peak = max([session.counts[n] for n in CLASS_NAMES] + [1])

    for i, (name, color, canon) in enumerate(CLASSES):
        x = margin + i * (card_w + gap)
        y = top
        cv2.rectangle(img, (x, y), (x + card_w, y + card_h), PANEL, -1)
        cv2.rectangle(img, (x, y), (x + card_w, y + card_h), (70, 82, 100), 1)
        cv2.rectangle(img, (x, y), (x + 6, y + card_h), color, -1)

        _text(img, name.upper(), x + 18, y + 28, 0.55, color, 1)
        count = session.counts[name]
        _text(img, str(count), x + 18, y + 82, 1.5, INK, 2)

        bar_x, bar_y, bar_w = x + 18, y + 98, card_w - 36
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 7),
                      (28, 34, 46), -1)
        fill = int(bar_w * count / peak)
        if fill:
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill, bar_y + 7),
                          color, -1)

        acc = rows.get(canon, {}).get("accuracy", 0.0)
        _text(img, f"AI: {delta.get(canon, 0)}  |  Acc: {acc:.1f}%",
              x + 18, y + 130, 0.45, MUTED, 1)

        btn_y, btn_h, btn_w = y + card_h - 44, 28, (card_w - 36 - 10) // 2
        _button(img, "-", x + 18, btn_y, btn_w, btn_h, "dec", name, fg=color)
        _button(img, "+", x + 18 + btn_w + 10, btn_y, btn_w, btn_h, "inc",
                name, fg=color)

    # ---- summary --------------------------------------------------
    sum_y = top + card_h + 14
    cv2.rectangle(img, (margin, sum_y), (W - margin, sum_y + 44), PANEL, -1)
    acc_total = report["weighted_accuracy"]
    acc_color = GREEN if report["passed"] else AMBER
    _text(img,
          f"MANUAL TOTAL: {session.manual_total}   |   "
          f"AI DELTA: {sum(delta.values())}   |   ACCURACY: {acc_total:.1f}%",
          margin + 16, sum_y + 28, 0.5, acc_color, 1)

    # ---- action bar -------------------------------------------------
    btn_y = sum_y + 56
    _button(img, "[SPACE] START/STOP", margin, btn_y, 220, 38, "toggle",
            fg=RED if session.state == "RUNNING" else GREEN,
            active=session.state == "RUNNING")
    _button(img, "[S] SAVE REPORT", margin + 232, btn_y, 190, 38, "save",
            fg=INK)
    _button(img, "[R] RESET", margin + 434, btn_y, 130, 38, "reset", fg=MUTED)

    toast = session.toast_text(now)
    hint = toast or "q/w/l +1   SHIFT -1   SPACE start/stop   N node   S save   ESC quit"
    _text(img, hint, margin, H - 14, 0.42, AMBER if toast else MUTED, 1)
    return img


# ---- Interactive app -------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Manual ground-truth counter with live AI accuracy sync")
    p.add_argument("--api-url", default="http://127.0.0.1:8000",
                   help="FastAPI base URL (default: %(default)s)")
    p.add_argument("--node", default=None,
                   help="node_id to sync against (default: first from API)")
    p.add_argument("--scenario", default="Live RTSP Validation",
                   help="Scenario label in the report (default: %(default)s)")
    p.add_argument("--output-dir", default="logs/benchmarks",
                   help="Where benchmark reports are written (default: %(default)s)")
    return p.parse_args(argv)


def handle_key(session: BenchmarkSession, key: int, output_dir) -> bool:
    """Apply one waitKey code. Returns False when the app should exit."""
    if key == 27:
        return False
    if key in KEYMAP_INC:
        session.increment(KEYMAP_INC[key])
    elif key in KEYMAP_DEC:
        session.decrement(KEYMAP_DEC[key])
    elif key in (ord("r"), ord("R")):
        session.reset()
    elif key == 32:
        session.toggle_session()
    elif key in (ord("n"), ord("N"), 9):
        session.cycle_node()
    elif key in (ord("s"), ord("S")):
        session.save_report(output_dir)
    return True


def handle_click(session: BenchmarkSession, x: int, y: int, output_dir):
    for (x0, y0, x1, y1), action, payload in UI_REGIONS:
        if x0 <= x <= x1 and y0 <= y <= y1:
            if action == "inc":
                session.increment(payload)
            elif action == "dec":
                session.decrement(payload)
            elif action == "toggle":
                session.toggle_session()
            elif action == "save":
                session.save_report(output_dir)
            elif action == "reset":
                session.reset()
            elif action == "cycle":
                session.cycle_node()
            return action
    return None


def run(session: BenchmarkSession, output_dir, fps: int = 50) -> int:
    """Blocking OpenCV loop. Kept out of import scope so tests stay headless."""
    session.poll_once()
    session.start_poller()
    win = "Manual Counter - Live Benchmark"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)
    cv2.setMouseCallback(
        win, lambda e, x, y, f, s: (
            e == cv2.EVENT_LBUTTONDOWN and handle_click(s, x, y, output_dir)),
        session)

    delay = max(1, int(1000 / fps))
    try:
        while True:
            cv2.imshow(win, render(session))
            key = cv2.waitKey(delay)
            if key == -1:
                # Window closed by the user: getWindowProperty raises.
                try:
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
                continue
            if not handle_key(session, key & 0xFF, output_dir):
                break
    finally:
        session.stop_poller()
        saved = session.maybe_save_on_exit(output_dir)
        if saved:
            print(f"Auto-saved benchmark: {saved}")
        cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    session = BenchmarkSession(api_url=args.api_url, node_id=args.node,
                               scenario=args.scenario)
    return run(session, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
