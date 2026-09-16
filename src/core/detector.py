"""RT-DETR inference + cross-class NMS. No tracking, no counting here."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import supervision as sv
from pathlib import Path


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = np.maximum(a[:2], b[:2])
    ix2, iy2 = np.minimum(a[2:], b[2:])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-6)


# COCO ids: 2 car, 3 motorcycle, 5 bus, 7 truck. mobil/bus/truk > motor.
CLASS_PRIORITY = {2: 2, 5: 2, 7: 2, 3: 1}


def cross_class_nms(dets: sv.Detections, duplicate_iou: float,
                    class_priority: Optional[dict] = None) -> sv.Detections:
    """Suppress overlapping cross-class boxes. Higher class priority wins;
    on a tie the bigger box (area-descending sweep) wins."""
    prio = class_priority or CLASS_PRIORITY
    xyxy, cls = dets.xyxy, dets.class_id
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    order = np.argsort(-areas)
    keep = np.ones(len(dets), dtype=bool)
    for ai in range(len(order)):
        i = order[ai]
        if not keep[i]:
            continue
        for bi in range(ai + 1, len(order)):
            j = order[bi]
            if not keep[j] or iou(xyxy[i], xyxy[j]) < duplicate_iou:
                continue
            if prio.get(int(cls[i]), 0) < prio.get(int(cls[j]), 0):
                keep[i] = False
                break
            keep[j] = False
    return dets[keep]


def centroids(xyxy: np.ndarray) -> np.ndarray:
    """(N,2) bounding-box centers."""
    return (xyxy[:, :2] + xyxy[:, 2:]) / 2.0


class Detector:
    """RT-DETR-L via ultralytics. Single conf floor for all classes so
    small/distant vehicles survive; higher imgsz preserves small-object detail."""

    def __init__(self, conf: float = 0.20, model_size: str = "rtdetr-l",
                 classes: Optional[Sequence[int]] = None, min_area: int = 20,
                 imgsz: int = 640, device: str = "cuda:0", half: bool = True,
                 duplicate_iou: float = 0.65) -> None:
        import torch
        from ultralytics import RTDETR

        self.conf = float(conf)
        self.classes = list(classes) if classes else [2, 3, 5, 7]
        self.min_area = int(min_area)
        self.imgsz = int(imgsz)
        self.device = device
        self.half = bool(half)
        self.duplicate_iou = float(duplicate_iou)

        path = str(model_size)
        if "." not in Path(path).name:
            path += ".pt"
        self.model = RTDETR(path)
        # Model-level FP16 weights (ultralytics >=8.4: predict kwarg half is
        # deprecated). TensorRT .engine is already built fp16: skip.
        if self.half and path.endswith(".pt") \
                and str(device).startswith("cuda") and torch.cuda.is_available():
            self.model.model.half()

    def detect(self, frame: np.ndarray) -> sv.Detections:
        r = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz,
            device=self.device, classes=self.classes,
            verbose=False,
        )[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return sv.Detections.empty()
        confs = b.conf.cpu().numpy()
        cls_ids = b.cls.int().cpu().numpy()
        mask = confs >= self.conf
        if self.min_area > 0:
            areas = (b.xyxy[:, 2] - b.xyxy[:, 0]) * (b.xyxy[:, 3] - b.xyxy[:, 1])
            mask &= areas.cpu().numpy() >= self.min_area
        if not mask.any():
            return sv.Detections.empty()
        dets = sv.Detections(
            xyxy=b.xyxy.cpu().numpy()[mask],
            confidence=confs[mask],
            class_id=cls_ids[mask],
        )
        return cross_class_nms(dets, self.duplicate_iou)
