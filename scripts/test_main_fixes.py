"""Integration check for the two main.py fixes (no GPU, no network):
1. no idle throttling  - AI worker consumes every frame the display delivers
2. per-node recording  - writer gets annotated frames even when unwatched,
                         `record` command toggles it off and finalizes the file
3. lazy rendering      - unwatched node skips display render + MJPEG publish
Run: python scripts/test_main_fixes.py
"""
import glob
import os
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main  # noqa: E402
from src.api import server as srv  # noqa: E402
from src.core.zone_counter import ZoneCounter  # noqa: E402

calls = {"detect": 0, "draw": 0, "sink": 0}
stop = threading.Event()


class FakeStreamer:
    def __init__(self):
        self._q = []

    def get(self, timeout=1.0):
        item = self._q.pop(0) if self._q else None
        if item is None:
            time.sleep(min(timeout, 0.05))
        return item

    def is_frozen(self):
        return False

    def effective_fps(self, default=25.0):
        return 25.0


class FakeDet:
    def detect(self, frame):
        calls["detect"] += 1
        return sv.Detections(
            xyxy=np.array([[100, 100, 140, 140]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([2], dtype=np.int32))


class FakeRenderer:
    def draw(self, *a, **k):
        calls["draw"] += 1


def sink(_out):
    calls["sink"] += 1


def feed(bus, streamer, n, delay_s):
    for _ in range(n):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        bus.publish_raw(frame, time.monotonic(), time.time())
        streamer._q.append((frame, time.monotonic(), time.time()))
        time.sleep(delay_s)


def main_test():
    tmpdir = tempfile.mkdtemp(prefix="atcs_test_")
    bus = main.FrameBus()
    streamer = FakeStreamer()
    zone = sv.PolygonZone(polygon=np.array(
        [[0, 0], [5000, 0], [5000, 5000], [0, 5000]], dtype=np.float32))
    counter = ZoneCounter([zone], class_labels={2: "mobil"},
                          min_displacement_px=0)
    prof = main.StageProfiler("test")
    reset_flag = {"v": False}
    t0 = time.monotonic()

    t_ai = threading.Thread(
        target=main._ai_worker,
        args=("node_test", bus, streamer, FakeDet(), counter,
              FakeRenderer(), threading.Lock(), stop, prof, reset_flag, t0,
              None, False, False, False, tmpdir, True),  # record_desired=True
        daemon=True)
    t_disp = threading.Thread(
        target=main._display_worker,
        args=("node_test", streamer, bus, FakeRenderer(), stop, prof, sink),
        kwargs={"focused": lambda _nid: False},  # simulates: nobody watching
        daemon=True)
    t_ai.start()
    t_disp.start()

    N = 45
    feed(bus, streamer, N, delay_s=0.02)  # ~50 fps feed
    time.sleep(0.8)

    # 1. AI consumed every frame (no throttle to 0.7 fps)
    print(f"fed={N} detect={calls['detect']}")
    assert calls["detect"] >= N - 2, f"throttled: {calls['detect']}/{N}"
    print("PASS no-idle-throttling")

    # 2. recording annotated frames while unwatched
    assert calls["draw"] >= N - 10, \
        f"record annotation missing: {calls['draw']}/{N}"
    print("PASS record-annotate-unwatched")

    # 3. lazy rendering: display never rendered/published for unwatched node
    assert calls["sink"] == 0, f"mjpeg publish on unwatched node: {calls['sink']}"
    print("PASS lazy-render-skips-publish")

    # 4. `record` command toggles OFF and finalizes the file
    srv.COMMAND_QUEUE.setdefault("node_test", []).append("record")
    time.sleep(1.0)
    vids = (glob.glob(os.path.join(tmpdir, "output_node_test_*.mp4"))
            + glob.glob(os.path.join(tmpdir, "output_node_test_*.avi")))
    assert vids, "no recording file created"
    cap = cv2.VideoCapture(vids[0])
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"recording={os.path.basename(vids[0])} frames={frames}")
    assert frames >= N - 10, f"recording too short: {frames}"
    print("PASS per-node-recording-toggle")

    stop.set()
    t_ai.join(timeout=5)
    t_disp.join(timeout=5)
    print("ALL MAIN FIX CHECKS PASS")


if __name__ == "__main__":
    try:
        main_test()
    finally:
        stop.set()
