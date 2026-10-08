import cv2
import numpy as np

GT_COLOR = (60, 210, 130)     # BGR green
PRED_COLOR = (200, 90, 220)   # BGR magenta


def overlay(gray, mask, color, alpha=0.55):
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    bgr[mask] = (1 - alpha) * bgr[mask] + alpha * np.array(color, np.float32)
    out = bgr.astype(np.uint8)
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnt, -1, color, 1)
    return out


def titled(img, text):
    bar = np.full((22, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def triptych(gray, gt, pred):
    """input | ground truth | prediction (with frame IoU)."""
    inter, uni = (gt & pred).sum(), (gt | pred).sum()
    iou = inter / uni if uni else 1.0
    gap = np.full((gray.shape[0] + 22, 6, 3), 255, np.uint8)
    return np.hstack([titled(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "input"), gap,
                      titled(overlay(gray, gt, GT_COLOR), "label"), gap,
                      titled(overlay(gray, pred, PRED_COLOR), f"prediction  IoU {iou:.2f}")])
