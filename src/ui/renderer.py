"""All OpenCV drawing: zones, boxes, trails, HUD. No detection/counting here."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

CLASS_INFO = {
    2: ("mobil", (0, 0, 255)),       # merah
    3: ("motor", (0, 255, 0)),       # hijau
    5: ("truk/bus", (0, 255, 255)),  # kuning
    7: ("truk/bus", (0, 255, 255)),  # kuning
}


class ZoneEditor:
    """Mouse vertex dragging on the display window. Sets .dirty for S-save."""
    GRAB_R = 14

    def __init__(self, zones, zone_cfgs):
        self.zones = zones
        self.zone_cfgs = zone_cfgs
        self.drag = None
        self.dirty = False

    def on_mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            best, bd = None, self.GRAB_R ** 2
            for zi, z in enumerate(self.zones):
                for vi, p in enumerate(z.polygon):
                    d = (p[0] - x) ** 2 + (p[1] - y) ** 2
                    if d < bd:
                        bd, best = d, (zi, vi)
            self.drag = best
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            zi, vi = self.drag
            self.zones[zi].polygon[vi] = np.array([x, y], dtype=np.float32)
            self.dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None


class Renderer:
    def __init__(self, zones: List[sv.PolygonZone], show_dims: bool = True,
                 debug: bool = False):
        self.show_dims = show_dims
        self.debug = debug
        self.zones = zones
        self.zone_anns = [sv.PolygonZoneAnnotator(
            zone=z, color=sv.Color(135, 206, 250), thickness=2,
            text_color=sv.Color.WHITE, text_scale=0.5, text_thickness=1,
            display_in_zone_count=False,
        ) for z in zones]

    def draw(self, frame: np.ndarray,
             snap: Optional[Tuple], counts: Dict[str, int],
             trails: Optional[dict] = None) -> None:
        """Mutates frame in place."""
        for ann in self.zone_anns:
            ann.annotate(scene=frame)
        for zone in self.zones:
            pts = np.asarray(zone.polygon, dtype=np.int32)
            for p in pts:
                cv2.circle(frame, tuple(p), 6, (255, 255, 255), 2)

        if self.debug and trails and snap is not None:
            for tid, trail in trails.items():
                pts = np.array(list(trail), dtype=np.int32)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False,
                                  (255, 0, 255), 1, cv2.LINE_AA)

        if snap is not None:
            xyxy, tids, clss, cents, confs = snap
            for i in range(len(cents)):
                cls = int(clss[i]) if clss is not None else -1
                info = CLASS_INFO.get(cls)
                color = info[1] if info else (255, 255, 255)
                lbl = info[0] if info else "?"
                x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"#{int(tids[i])} {lbl}"
                if self.debug and confs is not None:
                    text += f" {confs[i]:.2f}"
                if self.show_dims:
                    text += f" {x2-x1}x{y2-y1}"
                cv2.putText(frame, text, (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                cv2.circle(frame, tuple(cents[i]), 3, (0, 255, 255), -1)

        self._draw_counts(frame, counts)
        cv2.putText(frame, "drag zone | C=record | S=save | R=reset | Q=quit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_counts(frame, counts: Dict[str, int]):
        colors = {"motor": (0, 255, 0), "mobil": (0, 0, 255), "truk/bus": (0, 255, 255)}
        lines = [(f"TOTAL: {sum(counts.values())}", (255, 0, 255), 0.7, 10)]
        lines += [(f"{lbl}: {counts.get(lbl, 0)}", colors[lbl], 0.6, 2)
                  for lbl in ("motor", "mobil", "truk/bus")]

        x0, y0, pad, gap = 10, 28, 8, 26
        maxw = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, th)[0][0]
                   for t, _, s, th in lines)
        top = max(0, y0 - 22 - pad)
        bot = y0 + gap * (len(lines) - 1) + pad
        cv2.rectangle(frame, (x0 - pad, top), (x0 + maxw + pad, bot), (30, 30, 30), -1)

        y = y0
        for i, (text, color, scale, th) in enumerate(lines):
            if i:
                y += gap
            cv2.putText(frame, text, (x0, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)
