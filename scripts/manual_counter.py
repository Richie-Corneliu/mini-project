"""Manual counter for ground-truth validation. q=motor w=mobil l=bis/truk, r=reset, ESC=quit."""
import cv2
import numpy as np

LABELS = {"motor": (0, 255, 0), "mobil": (0, 0, 255), "bis/truk": (0, 255, 255)}
KEYMAP = {ord("q"): "motor", ord("w"): "mobil", ord("l"): "bis/truk"}

counts = dict.fromkeys(LABELS, 0)
W, H = 640, 200
FONT, SCALE, THICK = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2


def render():
    img = np.zeros((H, W, 3), np.uint8)
    parts = [f"{name}: {counts[name]}" for name in LABELS]
    parts.append(f"TOTAL: {sum(counts.values())}")
    colors = list(LABELS.values()) + [(0, 255, 255)]
    widths = [cv2.getTextSize(p, FONT, SCALE, THICK)[0][0] for p in parts]
    sep = cv2.getTextSize("  ", FONT, SCALE, THICK)[0][0]
    x = (W - (sum(widths) + sep * (len(parts) - 1))) // 2
    y = H // 2
    for text, color, w in zip(parts, colors, widths):
        cv2.putText(img, text, (x, y), FONT, SCALE, color, THICK, cv2.LINE_AA)
        x += w + sep
    hint = "q=motor  w=mobil  l=bis/truk  r=reset  ESC=quit"
    hw = cv2.getTextSize(hint, FONT, 0.45, 1)[0][0]
    cv2.putText(img, hint, ((W - hw) // 2, H - 16), FONT, 0.45,
                (160, 160, 160), 1, cv2.LINE_AA)
    return img


while True:
    cv2.imshow("Manual Counter", render())
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break
    if k in (ord("r"), ord("R")):
        counts = dict.fromkeys(LABELS, 0)
    elif k in KEYMAP:
        counts[KEYMAP[k]] += 1
cv2.destroyAllWindows()
