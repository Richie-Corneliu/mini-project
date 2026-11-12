import json

import numpy as np
import pytest
import supervision as sv

from scripts.benchmark_video import resolve_node_config, run_video_benchmark
from src.core.zone_counter import ZoneCounter


def test_resolve_node_config_merges_yaml_and_cli_overrides(tmp_path):
    settings = tmp_path / "settings.yaml"
    nodes = tmp_path / "nodes.yaml"
    settings.write_text("""detector:\n  conf: 0.2\n  car_min_area: 3000\n  truck_min_area: 9000\ncounter:\n  ghost_ttl_frames: 10\n  ghost_radius_px: 70\n  min_displacement_px: 30\nzones:\n- name: default\n  polygon: [[0, 0], [2, 0], [2, 2]]\n""", encoding="utf-8")
    nodes.write_text("""nodes:\n- id: cam-a\n  zones:\n  - name: override\n    polygon: [[1, 1], [9, 1], [9, 9]]\n  car_min_area: 4100\n  truck_min_area: 12000\n  ghost_ttl_frames: 7\n  min_displacement_px: 44\n""", encoding="utf-8")
    cfg = resolve_node_config("cam-a", settings_path=settings, nodes_path=nodes,
                              overrides={"car_min_area": 5000,
                                         "truck_min_area": 13000,
                                         "ghost_ttl_frames": 6})
    assert cfg["zones"][0]["name"] == "override"
    assert cfg["detector"]["car_min_area"] == 5000
    assert cfg["detector"]["truck_min_area"] == 13000
    assert cfg["counter"]["ghost_ttl_frames"] == 6
    assert cfg["counter"]["ghost_radius_px"] == 70
    assert cfg["counter"]["min_displacement_px"] == 44


class FakeDetector:
    conf = 0.2

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def detect(self, frame, **kwargs):
        self.calls += 1
        return next(self.outputs)


def _dets(box, cls=2, tid=1):
    return sv.Detections(xyxy=np.asarray([box], dtype=np.float32),
                         class_id=np.asarray([cls]),
                         confidence=np.asarray([0.9], dtype=np.float32),
                         tracker_id=np.asarray([tid]))


def test_run_video_benchmark_counts_handover_and_area_percentiles():
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [200, 0], [200, 200], [0, 200]],
                                           dtype=np.float32))
    counter = ZoneCounter([zone], class_labels={2: "mobil", 3: "motor"},
                          min_displacement_px=0, dedup_cooldown_sec=0)
    # Preserve controlled tracker IDs while exercising real ZoneCounter.update.
    counter.track = lambda dets, frame: dets
    outputs = [_dets([50, 50, 70, 70], tid=1),
               _dets([60, 60, 80, 80], tid=1),
               sv.Detections.empty(),
               _dets([65, 65, 85, 85], tid=2)]
    report = run_video_benchmark([np.zeros((220, 220, 3), np.uint8) for _ in outputs],
                                 FakeDetector(outputs), counter)
    assert report["frames_processed"] == 4
    assert report["counts"] == {"motor": 0, "mobil": 1, "truk_bus": 0}
    assert report["ghost_handovers"] == 1
    assert report["area_px_by_class"]["mobil"] == {
        "count": 3, "min": 400, "median": 400.0, "p90": 400.0, "max": 400}


def test_ground_truth_integration_and_report_files(tmp_path):
    zone = sv.PolygonZone(polygon=np.array([[0, 0], [200, 0], [200, 200], [0, 200]],
                                           dtype=np.float32))
    counter = ZoneCounter([zone], class_labels={2: "mobil"}, min_displacement_px=0)
    counter.track = lambda dets, frame: dets
    outputs = [_dets([50, 50, 70, 70], cls=2, tid=1)]
    report = run_video_benchmark([np.zeros((220, 220, 3), np.uint8)],
                                 FakeDetector(outputs), counter,
                                 gt={"motor": 0, "mobil": 1, "truk_bus": 0},
                                 scenario="synthetic", output_dir=tmp_path)
    assert report["accuracy"]["passed"] is True
    assert report["accuracy"]["weighted_accuracy"] == pytest.approx(100)
    json_path = next(tmp_path.glob("video_benchmark_*.json"))
    md_path = json_path.with_suffix(".md")
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["accuracy"]["passed"] is True
    assert "PASSED" in md_path.read_text(encoding="utf-8")
