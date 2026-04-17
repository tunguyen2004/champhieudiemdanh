"""
Mark scoring helpers for robust OMR decisions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def score_mark(roi_binary: np.ndarray, roi_gray: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Score a bubble ROI using multiple cues.
    """
    if roi_binary is None or roi_binary.size == 0:
        return {
            "fill_ratio": 0.0,
            "darkness": 0.0,
            "core_fill": 0.0,
            "ring_noise": 0.0,
            "score": 0.0,
        }

    h, w = roi_binary.shape[:2]
    fill_ratio = float(cv2.countNonZero(roi_binary) / roi_binary.size)

    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axis_x = max(int(w * 0.32), 1)
    axis_y = max(int(h * 0.32), 1)
    cv2.ellipse(mask, center, (axis_x, axis_y), 0, 0, 360, 255, -1)

    core_pixels = max(cv2.countNonZero(mask), 1)
    core_fill = float(cv2.countNonZero(cv2.bitwise_and(roi_binary, roi_binary, mask=mask)) / core_pixels)

    ring_pixels = max(roi_binary.size - core_pixels, 1)
    ring_nonzero = max(cv2.countNonZero(roi_binary) - int(core_fill * core_pixels), 0)
    ring_fill = ring_nonzero / ring_pixels
    ring_noise = max(0.0, ring_fill - core_fill * 0.6)

    if roi_gray is None or roi_gray.size == 0:
        darkness = core_fill
    else:
        mean_gray = cv2.mean(roi_gray, mask=mask)[0]
        darkness = float((255.0 - mean_gray) / 255.0)

    score = (
        0.55 * core_fill
        + 0.25 * fill_ratio
        + 0.20 * darkness
        - 0.25 * ring_noise
    )

    return {
        "fill_ratio": float(fill_ratio),
        "darkness": float(darkness),
        "core_fill": float(core_fill),
        "ring_noise": float(ring_noise),
        "score": float(score),
    }


def normalize_mark_scores(
    scores: Sequence[Dict[str, float]],
    temperature: float = 1.0
) -> List[float]:
    """
    Normalize raw mark scores within one local decision group (question/column).
    Returns probability-like values that sum to 1.
    """
    if not scores:
        return []

    raw = np.array([float(item.get("score", 0.0)) for item in scores], dtype=np.float32)
    if raw.size == 0:
        return []

    finite_mask = np.isfinite(raw)
    if not np.any(finite_mask):
        return [0.0] * int(raw.size)
    if not np.all(finite_mask):
        safe_floor = float(np.min(raw[finite_mask]))
        raw = np.where(finite_mask, raw, safe_floor).astype(np.float32)

    temp = max(float(temperature), 1e-6)
    shifted = raw - float(np.max(raw))
    logits = np.clip(shifted / temp, -50.0, 50.0)
    exp_vals = np.exp(logits)
    denom = float(np.sum(exp_vals))
    if denom <= 1e-12:
        return [1.0 / float(raw.size)] * int(raw.size)
    return (exp_vals / denom).astype(np.float32).tolist()


def choose_mark(
    scores: Sequence[Dict[str, float]],
    min_score: float = 0.55,
    min_gap: float = 0.08,
    normalize: bool = True,
    temperature: float = 1.0,
    multi_delta: float = 0.04,
) -> Tuple[List[int], float]:
    """
    Decide selected index/indices from score list.
    Returns (indices, normalized confidence).
    When `normalize=True`, confidence is computed inside one local group
    (for example one TF sub-question or one DG column).
    """
    if not scores:
        return [], 0.0

    raw = np.array([float(item.get("score", 0.0)) for item in scores], dtype=np.float32)
    if raw.size == 0:
        return [], 0.0

    if normalize:
        values = np.array(
            normalize_mark_scores(scores, temperature=temperature),
            dtype=np.float32
        )
    else:
        values = raw

    best_idx = int(np.argmax(values))
    best = float(values[best_idx])
    sorted_vals = sorted(values.tolist(), reverse=True)
    second = float(sorted_vals[1]) if len(sorted_vals) > 1 else 0.0
    gap = best - second

    if best < min_score:
        return [], max(0.0, min(1.0, best))

    # Very close peaks -> ambiguous / potentially multi.
    tie_window = max(float(multi_delta), float(min_gap) * 0.7)
    multi_indices = [idx for idx, value in enumerate(values.tolist()) if best - value <= tie_window]
    if len(multi_indices) > 1:
        return multi_indices, max(0.0, min(1.0, best))

    if normalize:
        confidence = max(0.0, min(1.0, best))
    else:
        confidence = max(0.0, min(1.0, (best - second + 0.02) / max(best + 1e-6, 1e-6)))

    return [best_idx], confidence
