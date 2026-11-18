"""VideoStreamer checks: per-frame gap carrying, jitter flag, no-drop backpressure.
Run:  python -m tests.test_video_streamer"""
import os
import threading
import time

from src.stream.video_streamer import VideoStreamer


def test_ffmpeg_capture_options_set():
    # Set at import so it applies to the first cv2.VideoCapture.
    assert "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ


def test_gap_is_per_frame_point_in_time():
    s = VideoStreamer("unused://", max_queue=4, gap_thresh_sec=0.5)
    t0 = time.monotonic()
    for i, gap in enumerate([None, 0.04, 0.8, 0.04]):
        s._queue.put((object(), t0 + i, t0 + i, gap))
    assert s.get(timeout=1) is not None
    assert not s.is_frozen()          # first frame, no gap
    s.get(timeout=1)
    assert not s.is_frozen()          # 40ms gap: normal
    s.get(timeout=1)
    assert s.is_frozen()              # 800ms gap: THIS frame is jitter
    s.get(timeout=1)
    assert not s.is_frozen()          # burst continues normal after the late frame
    assert s.frames_delivered == 4


def test_bounded_put_keeps_every_frame():
    # consumer stalls, queue fills, producer must block not drop
    s = VideoStreamer("unused://", max_queue=2)
    order = []

    def pump():
        for i in range(5):
            s._queue.put((i, float(i), float(i), None))
            order.append(i)

    th = threading.Thread(target=pump)
    th.start()
    time.sleep(0.2)
    assert th.is_alive()  # blocked on full queue, no drop
    got = [s.get(timeout=1)[0] for _ in range(5)]
    th.join(1)
    assert got == [0, 1, 2, 3, 4]
    assert not th.is_alive()


def test_unlimited_retries_when_max_retries_none_or_non_positive(monkeypatch):
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
    assert cfg["stream"]["max_retries"] is None
    assert VideoStreamer("unused://")._backoff_max == 30.0
    assert VideoStreamer("unused://", max_retries=0)._max_retries is None

    opens = {"count": 0}
    connected = threading.Event()

    class _FakeCap:
        def __init__(self, succeed):
            self._ok = succeed

        def isOpened(self):
            return self._ok

        def set(self, *a):
            pass

        def get(self, *a):
            return 25.0

        def read(self):
            connected.set()
            time.sleep(0.01)
            return False, None

        def release(self):
            pass

    def fake_capture(*args):
        opens["count"] += 1
        return _FakeCap(succeed=opens["count"] >= 8)

    monkeypatch.setattr("src.stream.video_streamer.cv2.VideoCapture", fake_capture)
    s = VideoStreamer("rtsp://cam/stream", backoff_min=0.001, backoff_max=0.005, max_retries=None)
    s.start()
    try:
        assert connected.wait(timeout=3.0), f"stopped early after {opens['count']} opens"
        assert opens["count"] >= 8
        assert s.connects >= 1
    finally:
        s.stop(join_timeout=2.0)


def test_redact_url_and_no_credential_leak_in_logs(caplog, monkeypatch):
    import logging
    from pathlib import Path
    import yaml
    from src.stream.video_streamer import redact_url

    cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
    assert "{user}:{pass}" in cfg["stream"]["source"]

    assert redact_url("rtsp://admin:Marktel12345@10.26.110.126/h264") == "rtsp://***:***@10.26.110.126/h264"
    assert redact_url("rtsp://admin:p@ss:123@10.26.110.126/h264") == "rtsp://***:***@10.26.110.126/h264"
    assert redact_url("rtsp://cam/stream") == "rtsp://cam/stream"
    assert redact_url(0) == "0"

    class _ClosedCap:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr("src.stream.video_streamer.cv2.VideoCapture", lambda *a: _ClosedCap())
    caplog.set_level(logging.WARNING, logger="src.stream.video_streamer")
    s = VideoStreamer(
        "rtsp://secretuser:SuperSecret99@10.0.0.1/ch1",
        backoff_min=0.001,
        backoff_max=0.002,
        max_retries=1,
    )
    s._run()
    assert "SuperSecret99" not in caplog.text
    assert "secretuser" not in caplog.text
    assert "***:***@10.0.0.1/ch1" in caplog.text


if __name__ == "__main__":
    test_ffmpeg_capture_options_set()
    print("PASS test_ffmpeg_capture_options_set")
    test_gap_is_per_frame_point_in_time()
    print("PASS test_gap_is_per_frame_point_in_time")
    test_bounded_put_keeps_every_frame()
    print("PASS test_bounded_put_keeps_every_frame")
