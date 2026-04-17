"""
Grid inference helpers for dynamic OMR layout detection.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


Point = Tuple[float, float]


def cluster_1d(values: Sequence[float], tolerance: float) -> Tuple[List[float], List[int]]:
    """
    Cluster 1D values by distance threshold and return (cluster_centers, assignments).
    """
    if not values:
        return [], []

    tolerance = max(float(tolerance), 1e-6)
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    clusters: List[List[Tuple[int, float]]] = [[indexed[0]]]

    for item in indexed[1:]:
        prev_val = clusters[-1][-1][1]
        if abs(item[1] - prev_val) <= tolerance:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    centers: List[float] = []
    assignments = [0] * len(values)
    for cluster_idx, cluster in enumerate(clusters):
        vals = [val for _, val in cluster]
        center = float(np.mean(vals))
        centers.append(center)
        for original_idx, _ in cluster:
            assignments[original_idx] = cluster_idx

    return centers, assignments


def infer_grid_signature(
    points: Sequence[Point],
    row_tolerance: float,
    col_tolerance: float
) -> Dict[str, object]:
    """
    Infer row/col structure for a set of points.
    """
    if not points:
        return {
            "rows": 0,
            "cols": 0,
            "row_centers": [],
            "col_centers": [],
            "occupancy": 0.0,
            "score": 0.0,
        }

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    row_centers, row_idx = cluster_1d(ys, row_tolerance)
    col_centers, col_idx = cluster_1d(xs, col_tolerance)

    rows = len(row_centers)
    cols = len(col_centers)
    if rows == 0 or cols == 0:
        return {
            "rows": 0,
            "cols": 0,
            "row_centers": row_centers,
            "col_centers": col_centers,
            "occupancy": 0.0,
            "score": 0.0,
        }

    occupancy_map = np.zeros((rows, cols), dtype=np.uint8)
    for r, c in zip(row_idx, col_idx):
        occupancy_map[r, c] = 1

    occupied = int(np.count_nonzero(occupancy_map))
    total = rows * cols
    occupancy = occupied / max(total, 1)

    # Score favors dense regular grids.
    score = occupancy * min(rows, cols) / max(rows, cols)

    return {
        "rows": rows,
        "cols": cols,
        "row_centers": row_centers,
        "col_centers": col_centers,
        "occupancy": float(occupancy),
        "score": float(score),
    }


def bbox_from_points(
    points: Sequence[Point],
    pad_x: float,
    pad_y: float,
    img_w: int,
    img_h: int
) -> Tuple[int, int, int, int]:
    """Compute clamped bbox from points with padding."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(int(min(xs) - pad_x), 0)
    y1 = max(int(min(ys) - pad_y), 0)
    x2 = min(int(max(xs) + pad_x), img_w - 1)
    y2 = min(int(max(ys) + pad_y), img_h - 1)
    return x1, y1, x2, y2

