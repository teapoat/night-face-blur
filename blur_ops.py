"""Ellipse face blur and a short box tracker. Parameters are the measured night blur."""

import cv2
import numpy as np


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0


class Track:
    def __init__(self, box):
        self.box = [float(v) for v in box[:4]]
        self.miss = 0


def update_tracks(tracks, dets):
    used = set()
    for track in tracks:
        best_i, best_v = -1, 0.2
        for i, det in enumerate(dets):
            if i in used:
                continue
            value = iou(track.box, det[:4])
            if value > best_v:
                best_v, best_i = value, i
        if best_i >= 0:
            used.add(best_i)
            det = dets[best_i]
            for j in range(4):
                track.box[j] = 0.6 * det[j] + 0.4 * track.box[j]
            track.miss = 0
        else:
            track.miss += 1
    tracks[:] = [track for track in tracks if track.miss <= 8]
    for i, det in enumerate(dets):
        if i not in used:
            tracks.append(Track(det))


def _odd_kernel(limit, frac):
    kernel = max(15, int(limit * frac) | 1)
    if kernel % 2 == 0:
        kernel += 1
    if kernel >= limit:
        kernel = limit if limit % 2 else limit - 1
    return kernel


def apply_blur(frame, box, pad=1.05, kernel_frac=0.5, passes=2, feather=15):
    height, width = frame.shape[:2]
    x, y, bw, bh = box[:4]
    cx, cy = x + bw / 2, y + bh / 2
    bw *= 1 + pad
    bh *= 1 + pad * 1.25
    x1 = int(max(0, cx - bw / 2))
    y1 = int(max(0, cy - bh / 2))
    x2 = int(min(width, cx + bw / 2))
    y2 = int(min(height, cy + bh / 2))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return
    roi = frame[y1:y2, x1:x2]
    limit = min(roi.shape[0], roi.shape[1])
    kernel = _odd_kernel(limit, kernel_frac)
    if kernel < 3:
        return
    blurred = roi
    for _ in range(max(1, passes)):
        blurred = cv2.GaussianBlur(blurred, (kernel, kernel), 0)
    mask = np.zeros(roi.shape[:2], np.float32)
    cv2.ellipse(
        mask,
        (roi.shape[1] // 2, roi.shape[0] // 2),
        (max(1, roi.shape[1] // 2 - 1), max(1, roi.shape[0] // 2 - 1)),
        0,
        0,
        360,
        1.0,
        -1,
    )
    feather_px = feather if min(roi.shape[:2]) > feather else 3
    if feather_px % 2 == 0:
        feather_px += 1
    mask = cv2.GaussianBlur(mask, (feather_px, feather_px), 0)[..., None]
    mixed = roi.astype(np.float32) * (1 - mask) + blurred.astype(np.float32) * mask
    frame[y1:y2, x1:x2] = mixed.astype(np.uint8)
