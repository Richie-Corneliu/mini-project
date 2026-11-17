"""RT-DETR inference + cross-class NMS. No tracking, no counting here."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import supervision as sv
from pathlib import Path

# COCO ids: 2 car, 3 motorcycle, 5 bus, 7 truck. mobil/bus/truk > motor.
CLASS_PRIORITY = {2: 2, 5: 2, 7: 2, 3: 1}


# Size heuristic: COCO's `truck` (7) and `bus` (5) heads swallow small pickups
# and box cars. A truck/bus box smaller than this many pixels is demoted to car
# (2). Pixels, not a frame ratio, so it lines up with min_area and the
# pixel-space zones/nodes config; operators tune it per camera perspective
# (detector.truck_min_area, per-node override in nodes.yaml).
CAR_CLS = 2
MOTOR_CLS = 3
TRUCK_CLS = (5, 7)
DEFAULT_TRUCK_MIN_AREA = 12000
DEFAULT_CAR_MIN_AREA = 3000
DEFAULT_SCALE_FAR = 0.42
DEFAULT_SCALE_NEAR = 1.95
MIN_PERSPECTIVE_SPAN_PX = 40.0


def _extract_polygon_y_bounds(polygons: Optional[Sequence]) -> Tuple[Optional[float], Optional[float]]:
    """Return (min_y, max_y) across PolygonZone objects, zone dicts, or point sequences."""
    if polygons is None:
        return None, None
    if isinstance(polygons, (sv.PolygonZone, dict)) or (
        isinstance(polygons, np.ndarray) and polygons.ndim == 2
    ):
        items = [polygons]
    else:
        items = polygons
    ys = []
    for item in items:
        pts = item.polygon if hasattr(item, "polygon") else (
            item.get("polygon") if isinstance(item, dict) else item
        )
        if pts is None:
            continue
        arr = np.asarray(pts, dtype=float)
        if arr.size == 0:
            continue
        arr = arr.reshape(-1, 2)
        ys.append(arr[:, 1])
    if not ys:
        return None, None
    all_y = np.concatenate(ys)
    return float(np.min(all_y)), float(np.max(all_y))


def reclassify_small_trucks(dets: sv.Detections,
                            truck_min_area: Union[float, np.ndarray],
                            truck_classes: Sequence[int] = TRUCK_CLS,
                            car_class: int = CAR_CLS) -> sv.Detections:
    """Demote truck/bus boxes with area < `truck_min_area` px to car, in place.

    One vectorised pass over the final post-NMS boxes (no per-box Python loop),
    negligible beside inference. Returns `dets` untouched when disabled
    (threshold <= 0) or empty.

    ponytail: box size alone cannot separate a far-away large truck from a
    near small pickup - both are small boxes. This is a per-camera heuristic
    with that known ceiling; a camera showing both at once needs a trained
    class head instead.
    """
    thr = np.asarray(truck_min_area, dtype=float)
    if dets.class_id is None or len(dets) == 0 or np.all(thr <= 0):
        return dets
    xyxy = dets.xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    demote = np.isin(dets.class_id, truck_classes) & (thr > 0) & (areas < thr)
    if demote.any():
        cls = dets.class_id.copy()
        cls[demote] = car_class
        dets.class_id = cls
        # Tag the demoted rows so downstream logic can tell them apart from a
        # genuine car. The flag rides in Detections.data and survives the
        # tracker's row-slice, so ZoneCounter can exempt these from the
        # cabin-suppression filter (a demoted truck is not a truck cabin).
        dets.data["reclassified"] = demote
    return dets


def reclassify_small_cars(dets: sv.Detections,
                          car_min_area: Union[float, np.ndarray],
                          car_class: int = CAR_CLS,
                          motor_class: int = MOTOR_CLS) -> sv.Detections:
    """Demote car boxes with area < `car_min_area` px to motorcycle, in place.

    A motorcycle turning sideways at an angle briefly widens its box into
    car-sized territory AND RT-DETR's attention flips the label to car; the
    flip is momentary but the resulting false car count is permanent. Small
    `car` boxes are far more often a sideways bike than a distant car — and a
    genuinely distant car will flicker between the two classes, which the
    counter's temporal voting absorbs. Same per-camera ceiling as the truck
    heuristic above.
    """
    thr = np.asarray(car_min_area, dtype=float)
    if dets.class_id is None or len(dets) == 0 or np.all(thr <= 0):
        return dets
    xyxy = dets.xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    demote = (dets.class_id == car_class) & (thr > 0) & (areas < thr)
    if demote.any():
        cls = dets.class_id.copy()
        cls[demote] = motor_class
        dets.class_id = cls
        # Tag the rows so ZoneCounter's cabin filter can still suppress a
        # small car box that is really a truck cabin — otherwise this sweep
        # would turn cabins into false motorcycle counts.
        dets.data["demoted_car"] = demote
    return dets


def apply_size_heuristics(dets: sv.Detections,
                          truck_min_area: Union[float, np.ndarray],
                          car_min_area: Union[float, np.ndarray]) -> sv.Detections:
    """Both tier demotions in one call. Car sweep first so a truck demoted to
    car by the second sweep lands on car (its final class) and is never
    re-demoted to motorcycle by the car threshold."""
    dets = reclassify_small_cars(dets, car_min_area)
    return reclassify_small_trucks(dets, truck_min_area)


def cross_class_nms(dets: sv.Detections,
                    duplicate_iou: float) -> sv.Detections:
    """Suppress overlapping cross-class boxes. Higher class priority wins;
    on a tie the bigger box (area-descending sweep) wins. The IoU matrix is
    computed once; the sweep is O(n²) over the kept candidates."""
    xyxy, cls = dets.xyxy, dets.class_id
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    order = np.argsort(-areas)
    iou_m = sv.box_iou_batch(xyxy, xyxy)
    keep = np.ones(len(dets), dtype=bool)
    for ai in range(len(order)):
        i = order[ai]
        if not keep[i]:
            continue
        for bi in range(ai + 1, len(order)):
            j = order[bi]
            if not keep[j] or iou_m[i, j] < duplicate_iou:
                continue
            if CLASS_PRIORITY.get(int(cls[i]), 0) < CLASS_PRIORITY.get(int(cls[j]), 0):
                keep[i] = False
                break
            keep[j] = False
    return dets[keep]


def centroids(xyxy: np.ndarray) -> np.ndarray:
    """(N,2) bounding-box centers."""
    return (xyxy[:, :2] + xyxy[:, 2:]) / 2.0


_HALF_TRUE = ("1", "true", "yes", "on")
_HALF_AUTO = ("", "auto", "none", "null")


def resolve_half(half, device: str = "cuda:0") -> bool:
    """Resolve the FP16 request into a real boolean.

    `None` / "auto" enables FP16 only when the device is CUDA *and* CUDA is
    genuinely available — half-precision weights on CPU or MPS either crash or
    silently fall back, so auto must never opt in there. An explicit value
    (bool or yaml string) always wins.
    """
    if isinstance(half, str):
        s = half.strip().lower()
        half = None if s in _HALF_AUTO else (s in _HALF_TRUE)
    if half is not None:
        return bool(half)
    try:
        import torch
    except Exception:
        return False
    return str(device).startswith("cuda") and bool(torch.cuda.is_available())


class Detector:
    """RT-DETR-L via ultralytics. Single conf floor for all classes so
    small/distant vehicles survive; higher imgsz preserves small-object detail."""

    def __init__(self, conf: float = 0.20, model_size: str = "rtdetr-l",
                 classes: Optional[Sequence[int]] = None, min_area: int = 20,
                 imgsz: int = 640, device: str = "cuda:0", half: Optional[bool] = None,
                 duplicate_iou: float = 0.65,
                 truck_min_area: float = DEFAULT_TRUCK_MIN_AREA,
                 car_min_area: float = DEFAULT_CAR_MIN_AREA,
                 perspective_scaling: bool = True,
                 y_ref_far: Optional[float] = None,
                 y_ref_mid: Optional[float] = None,
                 y_ref_near: Optional[float] = None,
                 scale_far: float = DEFAULT_SCALE_FAR,
                 scale_near: float = DEFAULT_SCALE_NEAR,
                 polygons: Optional[Sequence] = None) -> None:
        import torch
        from ultralytics import RTDETR

        self.conf = float(conf)
        self.classes = list(classes) if classes else [2, 3, 5, 7]
        self.min_area = int(min_area)
        self.imgsz = int(imgsz)
        self.device = device
        self.half = resolve_half(half, device)
        self.duplicate_iou = float(duplicate_iou)
        self.truck_min_area = float(truck_min_area)
        self.car_min_area = float(car_min_area)
        self.perspective_scaling = bool(perspective_scaling)
        self.y_ref_far = None if y_ref_far is None else float(y_ref_far)
        self.y_ref_near = None if y_ref_near is None else float(y_ref_near)
        self.y_ref_mid = None if y_ref_mid is None else float(y_ref_mid)
        self.scale_far = float(scale_far)
        self.scale_near = float(scale_near)
        if polygons is not None:
            self.configure_perspective_from_polygons(
                polygons,
                y_ref_far=self.y_ref_far,
                y_ref_mid=self.y_ref_mid,
                y_ref_near=self.y_ref_near,
            )
        elif (self.y_ref_mid is None and self.y_ref_far is not None
              and self.y_ref_near is not None):
            self.y_ref_mid = 0.5 * (self.y_ref_far + self.y_ref_near)

        path = str(model_size)
        if "." not in Path(path).name:
            path += ".pt"
        self.model = RTDETR(path)
        # Model-level FP16 weights (ultralytics >=8.4: predict kwarg half is
        # deprecated). TensorRT .engine is already built fp16: skip.
        if self.half and path.endswith(".pt") \
                and str(device).startswith("cuda") and torch.cuda.is_available():
            self.model.model.half()

    def configure_perspective_from_polygons(
        self,
        polygons: Optional[Sequence],
        y_ref_far: Optional[float] = None,
        y_ref_mid: Optional[float] = None,
        y_ref_near: Optional[float] = None,
        scale_far: Optional[float] = None,
        scale_near: Optional[float] = None,
        perspective_scaling: Optional[bool] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calibrate y_ref_far / y_ref_near from polygon vertex min/max Y."""
        poly_far, poly_near = _extract_polygon_y_bounds(polygons)
        if perspective_scaling is not None:
            self.perspective_scaling = bool(perspective_scaling)
        if scale_far is not None:
            self.scale_far = float(scale_far)
        if scale_near is not None:
            self.scale_near = float(scale_near)
        self.y_ref_far = float(y_ref_far) if y_ref_far is not None else poly_far
        self.y_ref_near = float(y_ref_near) if y_ref_near is not None else poly_near
        if y_ref_mid is not None:
            self.y_ref_mid = float(y_ref_mid)
        elif self.y_ref_far is not None and self.y_ref_near is not None:
            self.y_ref_mid = 0.5 * (self.y_ref_far + self.y_ref_near)
        else:
            self.y_ref_mid = None
        return self.y_ref_far, self.y_ref_mid, self.y_ref_near

    def _compute_perspective_scale(
        self,
        y: float,
        *,
        perspective_scaling: Optional[bool] = None,
        y_ref_far: Optional[float] = None,
        y_ref_mid: Optional[float] = None,
        y_ref_near: Optional[float] = None,
        scale_far: Optional[float] = None,
        scale_near: Optional[float] = None,
    ) -> float:
        """Center-anchored piecewise linear area scale along image Y."""
        ps = self.perspective_scaling if perspective_scaling is None else bool(perspective_scaling)
        far = self.y_ref_far if y_ref_far is None else float(y_ref_far)
        near = self.y_ref_near if y_ref_near is None else float(y_ref_near)
        if not ps or far is None or near is None or (near - far) < MIN_PERSPECTIVE_SPAN_PX:
            return 1.0
        mid_raw = self.y_ref_mid if y_ref_mid is None else float(y_ref_mid)
        if mid_raw is None:
            mid_raw = 0.5 * (far + near)
        y_mid = float(np.clip(mid_raw, far + 1.0, near - 1.0))
        sf = self.scale_far if scale_far is None else float(scale_far)
        sn = self.scale_near if scale_near is None else float(scale_near)
        yf = float(y)
        if yf <= y_mid:
            t = float(np.clip((yf - far) / (y_mid - far), 0.0, 1.0))
            return float(sf + t * (1.0 - sf))
        t = float(np.clip((yf - y_mid) / (near - y_mid), 0.0, 1.0))
        return float(1.0 + t * (sn - 1.0))

    def _apply_size_heuristics(
        self,
        dets: sv.Detections,
        truck_min_area: Optional[float] = None,
        car_min_area: Optional[float] = None,
        polygons: Optional[Sequence] = None,
        perspective_scaling: Optional[bool] = None,
        y_ref_far: Optional[float] = None,
        y_ref_mid: Optional[float] = None,
        y_ref_near: Optional[float] = None,
        scale_far: Optional[float] = None,
        scale_near: Optional[float] = None,
    ) -> sv.Detections:
        if dets.class_id is None or len(dets) == 0:
            return dets
        base_truck = self.truck_min_area if truck_min_area is None else float(truck_min_area)
        base_car = self.car_min_area if car_min_area is None else float(car_min_area)
        if base_truck <= 0 and base_car <= 0:
            return dets
        poly_far, poly_near = _extract_polygon_y_bounds(polygons) if polygons is not None else (None, None)
        far = float(y_ref_far) if y_ref_far is not None else (
            poly_far if polygons is not None else self.y_ref_far
        )
        near = float(y_ref_near) if y_ref_near is not None else (
            poly_near if polygons is not None else self.y_ref_near
        )
        if y_ref_mid is not None:
            mid = float(y_ref_mid)
        elif polygons is not None or y_ref_far is not None or y_ref_near is not None:
            mid = 0.5 * (far + near) if (far is not None and near is not None) else self.y_ref_mid
        else:
            mid = self.y_ref_mid
        ps = self.perspective_scaling if perspective_scaling is None else bool(perspective_scaling)
        if not ps or far is None or near is None or (near - far) < MIN_PERSPECTIVE_SPAN_PX:
            return apply_size_heuristics(dets, base_truck, base_car)
        scales = np.array([
            self._compute_perspective_scale(
                float(y2),
                perspective_scaling=ps,
                y_ref_far=far,
                y_ref_mid=mid,
                y_ref_near=near,
                scale_far=scale_far,
                scale_near=scale_near,
            )
            for y2 in dets.xyxy[:, 3]
        ], dtype=float)
        out = apply_size_heuristics(dets, base_truck * scales, base_car * scales)
        if out.class_id is not None and len(out):
            out.data["perspective_truck"] = np.isin(out.class_id, TRUCK_CLS)
        return out

    def detect(self, frame: np.ndarray,
               truck_min_area: Optional[float] = None,
               car_min_area: Optional[float] = None,
               polygons: Optional[Sequence] = None,
               perspective_scaling: Optional[bool] = None,
               y_ref_far: Optional[float] = None,
               y_ref_mid: Optional[float] = None,
               y_ref_near: Optional[float] = None,
               scale_far: Optional[float] = None,
               scale_near: Optional[float] = None) -> sv.Detections:
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
        dets = cross_class_nms(dets, self.duplicate_iou)
        return self._apply_size_heuristics(
            dets,
            truck_min_area=truck_min_area,
            car_min_area=car_min_area,
            polygons=polygons,
            perspective_scaling=perspective_scaling,
            y_ref_far=y_ref_far,
            y_ref_mid=y_ref_mid,
            y_ref_near=y_ref_near,
            scale_far=scale_far,
            scale_near=scale_near,
        )
