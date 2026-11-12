"""FP16 resolution (detector), per-track diagnostics (zone_counter) and the
toggleable diagnostic HUD (renderer). No GPU and no real model weights needed.
Run:  python -m pytest tests/test_diagnostics_and_fp16.py -q"""
import sys
import time

import numpy as np
import pytest
import supervision as sv

from src.core.detector import Detector, resolve_half
from src.core.zone_counter import ZoneCounter
from src.ui.renderer import Renderer, resolve_diagnostics

ZONE = np.array([[100, 400], [600, 400], [600, 700], [100, 700]], dtype=np.float32)
LABELS = {2: "mobil", 3: "motor", 5: "truk/bus", 7: "truk/bus"}


@pytest.fixture(autouse=True)
def _clean_diag_env(monkeypatch):
    """A stray ATCS_DIAGNOSTICS in the shell must not flip the default tests."""
    monkeypatch.delenv("ATCS_DIAGNOSTICS", raising=False)


def _dets(xyxy, cls, conf=None, tid=None):
    n = len(xyxy)
    return sv.Detections(
        xyxy=np.asarray(xyxy, dtype=np.float32),
        confidence=np.asarray(conf if conf is not None else [0.9] * n,
                              dtype=np.float32),
        class_id=np.asarray(cls, dtype=np.int32),
        tracker_id=np.asarray(tid, dtype=np.int64) if tid is not None else None)


def _counter(**kw):
    kw.setdefault("min_displacement_px", 0)
    return ZoneCounter([sv.PolygonZone(polygon=ZONE)], class_labels=LABELS, **kw)


# --------------------------------------------------------------------------- FP16

def test_resolve_half_auto_off_without_cuda():
    """auto/None must never opt into FP16 on CPU or MPS: half weights there
    either crash or silently fall back."""
    assert resolve_half(None, "cpu") is False
    assert resolve_half("auto", "cpu") is False
    assert resolve_half(None, "mps") is False
    assert resolve_half("AUTO", "mps") is False


def test_resolve_half_explicit_value_wins(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_half(False, "cuda:0") is False
    assert resolve_half("false", "cuda:0") is False
    assert resolve_half(True, "cpu") is True
    assert resolve_half("true", "cpu") is True


def test_resolve_half_auto_follows_cuda_availability(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_half(None, "cuda:0") is True
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_half(None, "cuda:0") is False


def test_resolve_half_survives_missing_torch(monkeypatch):
    """Import failure must degrade to False, not raise (headless boxes)."""
    monkeypatch.setitem(sys.modules, "torch", None)
    assert resolve_half(None, "cuda:0") is False


class _FakeModel:
    def __init__(self):
        self.half_calls = 0

    def half(self):
        self.half_calls += 1


class _FakeRTDETR:
    """Stands in for ultralytics.RTDETR; records every predict kwarg."""
    instances = []

    def __init__(self, path):
        self.path = path
        self.model = _FakeModel()
        self.predict_kwargs = []
        _FakeRTDETR.instances.append(self)

    def predict(self, frame, **kw):
        self.predict_kwargs.append(kw)

        class _R:
            boxes = None
        return [_R()]


def _detector(monkeypatch, cuda, **kw):
    monkeypatch.setattr("torch.cuda.is_available", lambda: cuda)
    monkeypatch.setattr("ultralytics.RTDETR", _FakeRTDETR)
    _FakeRTDETR.instances.clear()
    return Detector(**kw), _FakeRTDETR.instances[-1]


def test_detector_half_flag_and_weight_conversion(monkeypatch):
    det, fake = _detector(monkeypatch, cuda=True, device="cuda:0",
                          model_size="rtdetr-l")
    assert det.half is True and fake.model.half_calls == 1

    det, fake = _detector(monkeypatch, cuda=False, device="cpu",
                          model_size="rtdetr-l")
    assert det.half is False and fake.model.half_calls == 0

    det, fake = _detector(monkeypatch, cuda=True, device="cuda:0", half=False,
                          model_size="rtdetr-l")
    assert det.half is False and fake.model.half_calls == 0


def test_detector_engine_skips_half_conversion(monkeypatch):
    """TensorRT .engine is already built fp16: no weight conversion."""
    det, fake = _detector(monkeypatch, cuda=True, device="cuda:0",
                          model_size="rtdetr-l.engine")
    assert det.half is True and fake.model.half_calls == 0


def test_detector_predict_signature_backward_compatible(monkeypatch):
    """No `half` kwarg reaches predict (deprecated in ultralytics >=8.4), so
    existing mocks/stubs keep working and inference still returns Detections."""
    det, fake = _detector(monkeypatch, cuda=True, device="cuda:0",
                          model_size="rtdetr-l")
    out = det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert isinstance(out, sv.Detections) and len(out) == 0
    kw = fake.predict_kwargs[0]
    assert "half" not in kw
    assert set(kw) == {"conf", "imgsz", "device", "classes", "verbose"}


# ------------------------------------------------------------------- diagnostics

def test_track_diagnostics_values():
    zc = _counter()
    t = time.monotonic()
    ev = []
    for k, box in enumerate(([200, 450, 300, 550], [230, 470, 330, 570],
                             [260, 500, 360, 600])):
        ev += zc.update(_dets([box], [2], tid=[7]), now=t + k)
    assert len(ev) == 1 and zc.total == 1, ev          # counted once in zone 0

    d = zc.get_track_diagnostics(7, [260, 500, 360, 600])
    assert d["tid"] == 7
    assert d["area_px"] == 100 * 100
    assert d["displacement_px"] == 78.1                # (250,500) -> (310,550)
    assert d["car_locked"] is True                     # 3 car sightings
    assert d["motor_locked"] is False
    assert d["counted_zones"] == [0]
    assert d == zc.get_track_diagnostics(7, np.array([260, 500, 360, 600]))


def test_track_diagnostics_unknown_tid_and_read_only():
    zc = _counter()
    t = time.monotonic()
    zc.update(_dets([[200, 450, 300, 550]], [2], tid=[7]), now=t)
    before = (zc.total, dict(zc.total_by_class), dict(zc._start), dict(zc._prev),
              dict(zc.class_hist), set(zc._counted[0]), zc.n_frames)

    d = zc.get_track_diagnostics(999, [10, 10, 30, 50])
    assert d["area_px"] == 20 * 40
    assert d["displacement_px"] is None
    assert d["car_locked"] is False and d["motor_locked"] is False
    assert d["counted_zones"] == []
    assert zc.get_track_diagnostics(999)["area_px"] is None   # no bbox, no area

    after = (zc.total, dict(zc.total_by_class), dict(zc._start), dict(zc._prev),
             dict(zc.class_hist), set(zc._counted[0]), zc.n_frames)
    assert before == after                                   # never mutates


def test_motor_lock_and_diagnostics_summary():
    zc = _counter()
    t = time.monotonic()
    for k in range(3):
        zc.update(_dets([[200, 450 + k, 260, 550 + k]], [3], tid=[4]), now=t + k)
    d = zc.get_track_diagnostics(4, [200, 452, 260, 552])
    assert d["motor_locked"] is True and d["car_locked"] is False

    s = zc.diagnostics_summary()
    assert s["handovers"] == zc.ghost_handovers
    assert s["suppressed"] == zc.suppressed
    assert s["suppressed_cabin"] == zc.suppressed_cabin
    assert s["total"] == zc.total and s["frames"] == zc.n_frames


# ---------------------------------------------------------------------- renderer

def _snap(zc):
    xyxy = np.array([[260, 500, 360, 600]], dtype=np.float32)
    return (xyxy, np.array([7], dtype=np.int64), np.array([2], dtype=np.int32),
            np.array([[310, 550]], dtype=np.int32), np.array([0.9], dtype=np.float32))


def test_resolve_diagnostics_default_and_env_override(monkeypatch):
    assert resolve_diagnostics(None) is False
    assert resolve_diagnostics(False) is False
    assert resolve_diagnostics(True) is True
    assert resolve_diagnostics("false") is False
    assert resolve_diagnostics(None, env={"ATCS_DIAGNOSTICS": "1"}) is True
    assert resolve_diagnostics(None, env={"ATCS_DIAGNOSTICS": "0"}) is False
    # env beats config: flip the HUD on for one session without a redeploy
    assert resolve_diagnostics(False, env={"ATCS_DIAGNOSTICS": "true"}) is True

    monkeypatch.setenv("ATCS_DIAGNOSTICS", "1")
    r = Renderer([sv.PolygonZone(polygon=ZONE)])
    assert r.show_diagnostics is True
    monkeypatch.setenv("ATCS_DIAGNOSTICS", "0")
    assert Renderer([sv.PolygonZone(polygon=ZONE)]).show_diagnostics is False
    monkeypatch.delenv("ATCS_DIAGNOSTICS", raising=False)
    assert Renderer([sv.PolygonZone(polygon=ZONE)]).show_diagnostics is False


def test_render_clean_vs_diagnostic_output():
    zone = sv.PolygonZone(polygon=ZONE)
    zc = _counter()
    t = time.monotonic()
    for k, box in enumerate(([200, 450, 300, 550], [230, 470, 330, 570],
                             [260, 500, 360, 600])):
        zc.update(_dets([box], [2], tid=[7]), now=t + k)
    snap = _snap(zc)
    counts = dict(zc.total_by_class)

    clean = np.zeros((720, 1280, 3), dtype=np.uint8)
    Renderer([zone], counter=zc).draw(clean, snap, counts)      # default off
    diag = np.zeros((720, 1280, 3), dtype=np.uint8)
    Renderer([zone], counter=zc, show_diagnostics=True,
             car_min_area=3000, truck_min_area=10500).draw(diag, snap, counts)

    assert not np.array_equal(clean, diag)
    # top-right corner holds only the diagnostic strip (counts panel is top-left)
    assert clean[0:40, 1000:1280].sum() == 0
    assert diag[0:40, 1000:1280].sum() > 0


def test_render_without_counter_stays_clean():
    """show_diagnostics on but no counter wired: no crash, no telemetry."""
    zone = sv.PolygonZone(polygon=ZONE)
    r = Renderer([zone], show_diagnostics=True)
    assert r.show_diagnostics is True and r.counter is None
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    r.draw(frame, _snap(None), {"mobil": 1})
    assert frame[0:40, 1000:1280].sum() == 0
    r.draw(np.zeros((720, 1280, 3), dtype=np.uint8), None, {})


def test_diag_label_carries_telemetry():
    zc = _counter()
    t = time.monotonic()
    for k, box in enumerate(([200, 450, 300, 550], [230, 470, 330, 570],
                             [260, 500, 360, 600])):
        zc.update(_dets([box], [2], tid=[7]), now=t + k)
    r = Renderer([sv.PolygonZone(polygon=ZONE)], counter=zc, show_diagnostics=True)
    text = r._diag_label(7, "#7 mobil", [260, 500, 360, 600])
    assert text.startswith("#7 mobil | 10000px2 | d:78.1px"), text
    assert "[LOCK,CNT]" in text

    # uncounted, untracked id: telemetry still renders, no flags
    assert r._diag_label(42, "#42 motor", [10, 10, 30, 50]) == "#42 motor | 800px2"
