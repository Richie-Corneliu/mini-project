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


# --------------------------------------------------------- perspective scaling

def test_perspective_scale_unconfigured_disabled_and_thin_guard(monkeypatch):
    det_unconf, _ = _detector(monkeypatch, cuda=False, device="cpu", model_size="rtdetr-l")
    assert det_unconf._compute_perspective_scale(100.0) == 1.0
    assert det_unconf._compute_perspective_scale(500.0) == 1.0

    det_disabled, _ = _detector(monkeypatch, cuda=False, device="cpu", model_size="rtdetr-l",
                                perspective_scaling=False, y_ref_far=100.0, y_ref_near=500.0)
    assert det_disabled._compute_perspective_scale(100.0) == 1.0
    assert det_disabled._compute_perspective_scale(500.0) == 1.0

    det_thin, _ = _detector(monkeypatch, cuda=False, device="cpu", model_size="rtdetr-l",
                            y_ref_far=200.0, y_ref_near=239.0)
    assert det_thin._compute_perspective_scale(200.0) == 1.0
    assert det_thin._compute_perspective_scale(239.0) == 1.0


def test_perspective_scale_center_anchored_interpolation(monkeypatch):
    det, _ = _detector(monkeypatch, cuda=False, device="cpu", model_size="rtdetr-l",
                       y_ref_far=100.0, y_ref_near=500.0, scale_far=0.65, scale_near=1.25)
    assert det.y_ref_mid == pytest.approx(300.0)
    assert det._compute_perspective_scale(50.0) == pytest.approx(0.65)
    assert det._compute_perspective_scale(100.0) == pytest.approx(0.65)
    assert det._compute_perspective_scale(200.0) == pytest.approx(0.825)
    assert det._compute_perspective_scale(300.0) == pytest.approx(1.0)
    assert det._compute_perspective_scale(400.0) == pytest.approx(1.125)
    assert det._compute_perspective_scale(500.0) == pytest.approx(1.25)
    assert det._compute_perspective_scale(700.0) == pytest.approx(1.25)

    # y_ref_mid outside bounds is clamped inside (y_ref_far + 1, y_ref_near - 1)
    det_clamped, _ = _detector(monkeypatch, cuda=False, device="cpu", model_size="rtdetr-l",
                               y_ref_far=100.0, y_ref_mid=10.0, y_ref_near=500.0)
    assert det_clamped._compute_perspective_scale(101.0) == pytest.approx(1.0)


def test_configure_perspective_from_polygons(monkeypatch):
    poly_a = sv.PolygonZone(polygon=np.array([[100, 120], [300, 120], [300, 320], [100, 320]], dtype=np.float32))
    poly_b = sv.PolygonZone(polygon=np.array([[200, 250], [500, 250], [500, 620], [200, 620]], dtype=np.float32))

    det, _ = _detector(monkeypatch, cuda=False, device="cpu", model_size="rtdetr-l",
                       polygons=[poly_a, poly_b])
    assert det.y_ref_far == pytest.approx(120.0)
    assert det.y_ref_near == pytest.approx(620.0)
    assert det.y_ref_mid == pytest.approx(370.0)

    # Explicit override beats polygon min/max
    det.configure_perspective_from_polygons([poly_a], y_ref_far=80.0)
    assert det.y_ref_far == pytest.approx(80.0)
    assert det.y_ref_near == pytest.approx(320.0)
    assert det.y_ref_mid == pytest.approx(200.0)


class _StubBoxes:
    def __init__(self, xyxy, conf, cls_id):
        import torch
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.cls = torch.tensor(cls_id, dtype=torch.int64)

    def __len__(self):
        return len(self.xyxy)


def test_perspective_area_refinement_far_vs_near(monkeypatch):
    # Box 0 (far, y2=100): 44x50 = 2200 px2, raw cls=2 (car)
    #   car_min_area=3000 * 0.65 = 1950 -> stays car (2) with scaling, motor (3) without
    # Box 1 (near, y2=500): 120x100 = 12000 px2, raw cls=7 (truck)
    #   truck_min_area=10500 * 1.25 = 13125 -> demoted to car (2) with scaling, stays truck (7) without
    boxes = _StubBoxes(
        xyxy=[[100.0, 50.0, 144.0, 100.0], [200.0, 400.0, 320.0, 500.0]],
        conf=[0.9, 0.9],
        cls_id=[2, 7],
    )

    class _BoxModel(_FakeRTDETR):
        def predict(self, frame, **kw):
            self.predict_kwargs.append(kw)

            class _R:
                pass
            r = _R()
            r.boxes = boxes
            return [r]

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("ultralytics.RTDETR", _BoxModel)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    det_scaled = Detector(device="cpu", model_size="rtdetr-l",
                          car_min_area=3000, truck_min_area=10500,
                          y_ref_far=100.0, y_ref_near=500.0,
                          scale_far=0.65, scale_near=1.25,
                          perspective_scaling=True)
    out_scaled = det_scaled.detect(frame)
    assert list(out_scaled.class_id) == [2, 2]

    det_flat = Detector(device="cpu", model_size="rtdetr-l",
                        car_min_area=3000, truck_min_area=10500,
                        perspective_scaling=False)
    out_flat = det_flat.detect(frame)
    assert list(out_flat.class_id) == [3, 7]

    # Default Detector (truck_min_area=15000, scale_near=1.85 -> 27750 at y_ref_near)
    # demotes a large near-camera SUV / horizontal car (160x150 = 24000 px2) from truck (7) to car (2)
    suv_boxes = _StubBoxes(
        xyxy=[[300.0, 450.0, 460.0, 600.0]],
        conf=[0.92],
        cls_id=[7],
    )
    boxes.xyxy, boxes.conf, boxes.cls = suv_boxes.xyxy, suv_boxes.conf, suv_boxes.cls
    det_default = Detector(device="cpu", model_size="rtdetr-l",
                           y_ref_far=150.0, y_ref_near=600.0)
    assert det_default.scale_near == pytest.approx(1.85)
    assert det_default.truck_min_area == pytest.approx(15000.0)
    out_suv = det_default.detect(frame)
    assert list(out_suv.class_id) == [2]
