"""VideoStreamer checks: per-frame gap carrying, jitter flag, no-drop backpressure.
Run:  python -m tests.test_video_streamer"""
import threading
import time

from src.stream.video_streamer import VideoStreamer


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


if __name__ == "__main__":
    test_gap_is_per_frame_point_in_time()
    print("PASS test_gap_is_per_frame_point_in_time")
    test_bounded_put_keeps_every_frame()
    print("PASS test_bounded_put_keeps_every_frame")
