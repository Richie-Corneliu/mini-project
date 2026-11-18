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

    # Default Detector (truck_min_area=12000, scale_far=0.42 -> 5040, scale_near=1.95 -> 23400)
    # - far medium truck (80x70 = 5600 px2 at y2=150) stays truck (7) with perspective_truck=True
    # - near large SUV / horizontal car (150x150 = 22500 px2 at y2=600) is demoted to car (2)
    suv_boxes = _StubBoxes(
        xyxy=[[100.0, 80.0, 180.0, 150.0], [300.0, 450.0, 450.0, 600.0]],
        conf=[0.90, 0.92],
        cls_id=[7, 7],
    )
    boxes.xyxy, boxes.conf, boxes.cls = suv_boxes.xyxy, suv_boxes.conf, suv_boxes.cls
    det_default = Detector(device="cpu", model_size="rtdetr-l",
                           y_ref_far=150.0, y_ref_near=600.0)
    assert det_default.scale_far == pytest.approx(0.42)
    assert det_default.scale_near == pytest.approx(1.95)
    assert det_default.truck_min_area == pytest.approx(12000.0)
    out_suv = det_default.detect(frame)
    assert list(out_suv.class_id) == [7, 2]
    assert list(out_suv.data["perspective_truck"]) == [True, False]


def test_truck_min_height_ratio_scales_across_resolutions(monkeypatch):
    """7% frame-height guard demotes wide flat SUVs/glare on both 720p and 1080p streams."""
    state = {"xyxy": [[100.0, 300.0, 220.0, 348.0], [300.0, 290.0, 420.0, 345.0]]}

    class _HeightModel(_FakeRTDETR):
        def predict(self, frame, **kw):
            self.predict_kwargs.append(kw)
            n = len(state["xyxy"])

            class _R:
                boxes = _StubBoxes(state["xyxy"], [0.9] * n, [7] * n)

            return [_R()]

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("ultralytics.RTDETR", _HeightModel)

    det = Detector(
        device="cpu", model_size="rtdetr-l",
        truck_min_area=1000, car_min_area=100,
        truck_min_height_ratio=0.070,
        y_ref_far=100.0, y_ref_near=600.0,  # mid = 350 (scale ~ 1.0)
    )
    # 720p: base_h = 50.4 px -> h=48 demoted to car (2), h=55 stays truck (7)
    out_720 = det.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert list(out_720.class_id) == [2, 7]
    assert list(out_720.data["perspective_truck"]) == [False, True]

    # 1080p: base_h = 75.6 px -> h=70 demoted to car (2), h=80 stays truck (7)
    state["xyxy"] = [[100.0, 280.0, 260.0, 350.0], [300.0, 270.0, 460.0, 350.0]]
    out_1080 = det.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert list(out_1080.class_id) == [2, 7]
    assert list(out_1080.data["perspective_truck"]) == [False, True]

    # Side-view override (0.040 -> 28.8 px on 720p) allows h=48 box to stay truck
    state["xyxy"] = [[100.0, 300.0, 220.0, 348.0]]
    out_side = det.detect(np.zeros((720, 1280, 3), dtype=np.uint8), truck_min_height_ratio=0.040)
    assert list(out_side.class_id) == [7]


def test_hot_reloader_updates_detector_and_counter(tmp_path, monkeypatch):
    import os
    from main import ConfigHotReloader

    settings_file = tmp_path / "settings.yaml"
    nodes_file = tmp_path / "nodes.yaml"
    settings_file.write_text(
        "detector:\n  conf: 0.2\n  car_min_area: 3000\n  truck_min_area: 12000\n"
        "  truck_min_height_ratio: 0.070\n  scale_far: 0.42\n  scale_near: 1.95\n"
        "counter:\n  min_displacement_px: 12\n  flow_max_angle_deg: 75\n"
        "  ghost_ttl_frames: 6\n  ghost_radius_px: 35\n  truck_confirm_frames: 5\n",
        encoding="utf-8",
    )
    nodes_file.write_text(
        "nodes:\n- id: node_01\n  truck_min_area: 12000\n",
        encoding="utf-8",
    )

    det, _ = _detector(monkeypatch, cuda=False, device="cpu", model_size="rtdetr-l",
                       truck_min_area=12000, car_min_area=3000, scale_far=0.42, scale_near=1.95)
    zc = _counter(min_displacement_px=12, ghost_ttl_frames=6, ghost_radius_px=35)
    det_over = {"truck_min_area": 12000.0}

    reloader = ConfigHotReloader(settings_path=settings_file, nodes_path=nodes_file,
                                 det=det, check_interval_sec=2.0)
    reloader.register_node("node_01", zc, det_overrides=det_over)

    # First check initializes _last_check without reloading unchanged files
    assert reloader.check_and_reload(now=10.0) is False

    # Modify YAML files and advance mtime
    settings_file.write_text(
        "detector:\n  conf_threshold: 0.28\n  car_min_area: 2500\n  truck_min_area: 11000\n"
        "  truck_min_height_ratio: 0.065\n  scale_far: 0.38\n  scale_near: 2.1\n"
        "counter:\n  min_displacement_px: 18\n  flow_max_angle_deg: 60\n"
        "  ghost_ttl_frames: 8\n  ghost_radius_px: 40\n  truck_confirm_frames: 6\n",
        encoding="utf-8",
    )
    nodes_file.write_text(
        "nodes:\n- id: node_01\n  truck_min_area: 13500\n  truck_min_height_ratio: 0.050\n"
        "  truck_confirm_frames: 4\n  allowed_directions: [\"down\"]\n",
        encoding="utf-8",
    )
    future_mtime = os.path.getmtime(settings_file) + 5.0
    os.utime(settings_file, (future_mtime, future_mtime))
    os.utime(nodes_file, (future_mtime, future_mtime))

    # Before 2.0s interval: skipped
    assert reloader.check_and_reload(now=11.0) is False
    # After 2.0s interval: reloaded in-place
    assert reloader.check_and_reload(now=12.5) is True
    assert det.conf == pytest.approx(0.28)
    assert det.car_min_area == pytest.approx(2500.0)
    assert det.truck_min_area == pytest.approx(11000.0)
    assert det.truck_min_height_ratio == pytest.approx(0.065)
    assert det.scale_far == pytest.approx(0.38)
    assert det.scale_near == pytest.approx(2.1)
    assert det_over["truck_min_area"] == pytest.approx(13500.0)
    assert det_over["truck_min_height_ratio"] == pytest.approx(0.050)
    assert zc.min_disp2 == pytest.approx(18.0 ** 2)
    assert zc.allowed_directions == ["down"]
    assert zc.flow_max_angle_deg == pytest.approx(60.0)
    assert zc.ghost_ttl_frames == 8
    assert zc.ghost_r2 == pytest.approx(40.0 ** 2)
    assert zc.truck_confirm_frames == 4
    assert zc.truck_min_height_ratio == pytest.approx(0.050)
