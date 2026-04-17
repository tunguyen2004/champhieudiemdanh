"""
Dynamic layout analysis for OMR sheets.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .grid_inference import bbox_from_points, infer_grid_signature


Candidate = Dict[str, float]
Component = Dict[str, object]
BBox = Tuple[int, int, int, int]


def _contour_circularity(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return 0.0
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0:
        return 0.0
    return float((4.0 * np.pi * area) / (perimeter * perimeter))


def _estimate_diameter(candidates: Sequence[Candidate]) -> float:
    if not candidates:
        return 8.0
    areas = [max(float(item["area"]), 1.0) for item in candidates]
    diameters = [np.sqrt((4.0 * area) / np.pi) for area in areas]
    return float(np.median(diameters))


def _deduplicate_candidates(candidates: Sequence[Candidate], merge_dist: float) -> List[Candidate]:
    if not candidates:
        return []

    merge_dist_sq = max(merge_dist, 1.0) ** 2
    accepted: List[Candidate] = []

    for cand in sorted(candidates, key=lambda item: float(item.get("area", 0.0)), reverse=True):
        merged = False
        for keep in accepted:
            dx = float(cand["cx"]) - float(keep["cx"])
            dy = float(cand["cy"]) - float(keep["cy"])
            if (dx * dx) + (dy * dy) > merge_dist_sq:
                continue

            keep_area = max(float(keep.get("area", 1.0)), 1.0)
            cand_area = max(float(cand.get("area", 1.0)), 1.0)
            total = keep_area + cand_area
            keep["cx"] = float((keep["cx"] * keep_area + cand["cx"] * cand_area) / total)
            keep["cy"] = float((keep["cy"] * keep_area + cand["cy"] * cand_area) / total)
            keep["area"] = float(max(keep_area, cand_area))
            keep["circularity"] = float(max(float(keep.get("circularity", 0.0)), float(cand.get("circularity", 0.0))))

            keep_x1 = min(float(keep["x"]), float(cand["x"]))
            keep_y1 = min(float(keep["y"]), float(cand["y"]))
            keep_x2 = max(float(keep["x"] + keep["w"]), float(cand["x"] + cand["w"]))
            keep_y2 = max(float(keep["y"] + keep["h"]), float(cand["y"] + cand["h"]))
            keep["x"] = keep_x1
            keep["y"] = keep_y1
            keep["w"] = keep_x2 - keep_x1
            keep["h"] = keep_y2 - keep_y1
            merged = True
            break

        if not merged:
            accepted.append(dict(cand))

    return accepted


def _detect_hough_candidates(
    grayscale: np.ndarray,
    diameter_hint: float,
    dyn_cfg: Dict[str, object],
    min_area: float,
    max_area: float,
) -> List[Candidate]:
    blur_ksize = int(dyn_cfg.get("hough_blur_ksize", 5))
    if blur_ksize < 3:
        blur_ksize = 3
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    work = cv2.medianBlur(grayscale, blur_ksize)

    h, w = grayscale.shape[:2]
    if diameter_hint <= 0:
        diameter_hint = max(8.0, min(h, w) * 0.01)

    min_dist = float(dyn_cfg.get("hough_min_dist", max(8.0, diameter_hint * 0.82)))
    param1 = float(dyn_cfg.get("hough_param1", 80.0))
    param2 = float(dyn_cfg.get("hough_param2", 16.0))
    min_radius = int(round(float(dyn_cfg.get("hough_min_radius", max(3.0, diameter_hint * 0.33)))))
    max_radius = int(round(float(dyn_cfg.get("hough_max_radius", max(float(min_radius + 1), diameter_hint * 0.95)))))
    if max_radius <= min_radius:
        max_radius = min_radius + 2

    circles = cv2.HoughCircles(
        work,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []

    candidates: List[Candidate] = []
    for circle in circles[0]:
        cx, cy, r = float(circle[0]), float(circle[1]), float(circle[2])
        if r <= 0:
            continue
        area = float(np.pi * r * r)
        if area < (min_area * 0.45) or area > (max_area * 1.75):
            continue

        x = cx - r
        y = cy - r
        d = r * 2.0
        candidates.append(
            {
                "cx": cx,
                "cy": cy,
                "x": x,
                "y": y,
                "w": d,
                "h": d,
                "area": area,
                "circularity": 1.0,
            }
        )

    return candidates


def detect_bubble_candidates(
    binary: np.ndarray,
    grayscale: Optional[np.ndarray] = None,
    config: Optional[Dict[str, object]] = None,
) -> Tuple[List[Candidate], Dict[str, int]]:
    """
    Detect candidate bubble centers from contour + optional Hough circles.
    """
    h, w = binary.shape[:2]
    img_area = float(h * w)

    dyn_cfg = (config or {}).get("dynamic_layout", {})
    min_area = float(dyn_cfg.get("bubble_min_area", max(30.0, img_area * 0.000006)))
    max_area = float(dyn_cfg.get("bubble_max_area", img_area * 0.0018))
    min_circularity = float(dyn_cfg.get("bubble_min_circularity", 0.45))
    min_aspect = float(dyn_cfg.get("bubble_min_aspect", 0.55))
    max_aspect = float(dyn_cfg.get("bubble_max_aspect", 1.75))

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_candidates: List[Candidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        if cw <= 1 or ch <= 1:
            continue

        aspect = cw / max(float(ch), 1e-6)
        if aspect < min_aspect or aspect > max_aspect:
            continue

        circularity = _contour_circularity(contour)
        if circularity < min_circularity:
            continue

        contour_candidates.append(
            {
                "cx": float(x + (cw / 2.0)),
                "cy": float(y + (ch / 2.0)),
                "x": float(x),
                "y": float(y),
                "w": float(cw),
                "h": float(ch),
                "area": area,
                "circularity": circularity,
            }
        )

    diameter_hint = _estimate_diameter(contour_candidates)
    hough_candidates: List[Candidate] = []
    if grayscale is not None and bool(dyn_cfg.get("hough_enabled", True)):
        hough_candidates = _detect_hough_candidates(
            grayscale,
            diameter_hint=diameter_hint,
            dyn_cfg=dyn_cfg,
            min_area=min_area,
            max_area=max_area,
        )

    merge_dist = float(dyn_cfg.get("candidate_merge_dist", max(3.0, diameter_hint * 0.55)))
    merged = _deduplicate_candidates(contour_candidates + hough_candidates, merge_dist=merge_dist)
    stats = {
        "contour": len(contour_candidates),
        "hough": len(hough_candidates),
        "merged": len(merged),
    }
    return merged, stats


def _connected_components(
    candidates: Sequence[Candidate],
    max_dx: float,
    max_dy: float,
    max_dist: float,
) -> List[List[int]]:
    n = len(candidates)
    visited = [False] * n
    components: List[List[int]] = []
    max_dist_sq = max(max_dist, 1.0) ** 2

    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        group = [start]

        while stack:
            i = stack.pop()
            c1 = candidates[i]
            for j in range(n):
                if visited[j]:
                    continue
                c2 = candidates[j]
                dx = float(c1["cx"]) - float(c2["cx"])
                dy = float(c1["cy"]) - float(c2["cy"])
                if abs(dx) > max_dx or abs(dy) > max_dy:
                    continue
                if (dx * dx) + (dy * dy) > max_dist_sq:
                    continue
                visited[j] = True
                stack.append(j)
                group.append(j)

        components.append(group)

    return components


def _bbox_center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))


def _component_to_model(
    component_id: int,
    indices: Sequence[int],
    candidates: Sequence[Candidate],
    diameter: float,
    img_w: int,
    img_h: int,
    min_points: int,
    row_cluster_scale: float,
    col_cluster_scale: float,
) -> Optional[Component]:
    if len(indices) < min_points:
        return None

    points = [(float(candidates[idx]["cx"]), float(candidates[idx]["cy"])) for idx in indices]
    grid = infer_grid_signature(
        points,
        row_tolerance=max(3.0, diameter * row_cluster_scale),
        col_tolerance=max(3.0, diameter * col_cluster_scale),
    )

    rows = int(grid["rows"])
    cols = int(grid["cols"])
    if rows < 2 or cols < 2:
        return None

    bbox = bbox_from_points(
        points,
        pad_x=max(2.0, diameter * 1.05),
        pad_y=max(2.0, diameter * 1.05),
        img_w=img_w,
        img_h=img_h,
    )
    cx, cy = _bbox_center(bbox)
    bw = max(1, int(bbox[2] - bbox[0] + 1))
    bh = max(1, int(bbox[3] - bbox[1] + 1))

    return {
        "id": component_id,
        "rows": rows,
        "cols": cols,
        "count": len(indices),
        "occupancy": float(grid["occupancy"]),
        "score": float(grid["score"]),
        "bbox": bbox,
        "cx": cx,
        "cy": cy,
        "bw": bw,
        "bh": bh,
        "points": points,
    }


def _component_match_score(component: Component, target_rows: int, target_cols: int) -> float:
    row_diff = abs(int(component["rows"]) - target_rows)
    col_diff = abs(int(component["cols"]) - target_cols)
    occupancy = float(component["occupancy"])
    quality_bonus = float(component["score"])
    density_penalty = max(0.0, 0.82 - occupancy) * 2.0
    return (row_diff * 2.2) + (col_diff * 2.2) + density_penalty - (quality_bonus * 2.0)


def _region_to_bbox(region: Dict[str, object], img_w: int, img_h: int) -> BBox:
    x1 = int(round(float(region["x1"]) * img_w))
    y1 = int(round(float(region["y1"]) * img_h))
    x2 = int(round(float(region["x2"]) * img_w))
    y2 = int(round(float(region["y2"]) * img_h))
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    if x2 <= x1:
        x2 = min(img_w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h - 1, y1 + 1)
    return x1, y1, x2, y2


def _normalize_region(bbox: BBox, rows: int, cols: int, img_w: int, img_h: int) -> Dict[str, float]:
    x1, y1, x2, y2 = bbox
    return {
        "x1": max(0.0, x1 / max(img_w, 1)),
        "y1": max(0.0, y1 / max(img_h, 1)),
        "x2": min(1.0, x2 / max(img_w, 1)),
        "y2": min(1.0, y2 / max(img_h, 1)),
        "rows": int(rows),
        "cols": int(cols),
    }


def _bbox_overlap_ratio(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    aw = max(1, ax2 - ax1)
    ah = max(1, ay2 - ay1)
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    union = (aw * ah) + (bw * bh) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _pick_single_component(
    models: Sequence[Component],
    used_ids: set,
    template_region: Dict[str, object],
    target_rows: int,
    target_cols: int,
    row_tol: int,
    col_tol: int,
    img_w: int,
    img_h: int,
) -> Optional[Component]:
    expected_bbox = _region_to_bbox(template_region, img_w, img_h)
    exp_cx, exp_cy = _bbox_center(expected_bbox)
    exp_w = max(1, int(expected_bbox[2] - expected_bbox[0]))
    exp_h = max(1, int(expected_bbox[3] - expected_bbox[1]))
    x_tol = max(16.0, exp_w * 0.70)
    y_tol = max(16.0, exp_h * 0.70)

    best_component: Optional[Component] = None
    best_score: Optional[float] = None
    for comp in models:
        if comp["id"] in used_ids:
            continue
        row_diff = abs(int(comp["rows"]) - target_rows)
        col_diff = abs(int(comp["cols"]) - target_cols)
        if row_diff > row_tol or col_diff > col_tol:
            continue

        dx = abs(float(comp["cx"]) - exp_cx)
        dy = abs(float(comp["cy"]) - exp_cy)
        if dx > (x_tol * 1.7) or dy > (y_tol * 1.7):
            continue

        overlap = _bbox_overlap_ratio(comp["bbox"], expected_bbox)
        score = _component_match_score(comp, target_rows, target_cols)
        score += (dx / max(x_tol, 1.0)) * 1.4
        score += (dy / max(y_tol, 1.0)) * 1.4
        score += (1.0 - overlap) * 2.2

        if best_score is None or score < best_score:
            best_component = comp
            best_score = score

    if best_component is not None:
        used_ids.add(int(best_component["id"]))
    return best_component


def _assign_components_to_groups(
    models: Sequence[Component],
    used_ids: set,
    template_groups: Sequence[Dict[str, object]],
    target_rows: int,
    target_cols_fn,
    row_tol: int,
    col_tol: int,
    img_w: int,
    img_h: int,
    max_score: float,
) -> List[Optional[Component]]:
    assigned: List[Optional[Component]] = [None] * len(template_groups)
    local_used: set = set()

    for idx, template in enumerate(template_groups):
        expected_bbox = _region_to_bbox(template, img_w, img_h)
        exp_cx, exp_cy = _bbox_center(expected_bbox)
        exp_w = max(1, int(expected_bbox[2] - expected_bbox[0]))
        exp_h = max(1, int(expected_bbox[3] - expected_bbox[1]))
        x_tol = max(10.0, exp_w * 0.95)
        y_tol = max(10.0, exp_h * 0.95)
        target_cols = int(target_cols_fn(template))

        best_component: Optional[Component] = None
        best_local_score: Optional[float] = None
        for comp in models:
            comp_id = int(comp["id"])
            if comp_id in used_ids or comp_id in local_used:
                continue

            row_diff = abs(int(comp["rows"]) - target_rows)
            col_diff = abs(int(comp["cols"]) - target_cols)
            if row_diff > row_tol or col_diff > col_tol:
                continue

            dx = abs(float(comp["cx"]) - exp_cx)
            dy = abs(float(comp["cy"]) - exp_cy)
            if dx > (x_tol * 1.75) or dy > (y_tol * 1.75):
                continue

            overlap = _bbox_overlap_ratio(comp["bbox"], expected_bbox)
            size_penalty = abs(float(comp["bw"]) - exp_w) / exp_w
            size_penalty += abs(float(comp["bh"]) - exp_h) / exp_h

            local_score = _component_match_score(comp, target_rows, target_cols)
            local_score += (dx / max(x_tol, 1.0)) * 1.35
            local_score += (dy / max(y_tol, 1.0)) * 1.60
            local_score += (1.0 - overlap) * 1.90
            local_score += size_penalty

            if best_local_score is None or local_score < best_local_score:
                best_component = comp
                best_local_score = local_score

        if best_component is not None and best_local_score is not None and best_local_score <= max_score:
            assigned[idx] = best_component
            local_used.add(int(best_component["id"]))

    for comp_id in local_used:
        used_ids.add(comp_id)
    return assigned


def _region_center_px(region: Dict[str, object], img_w: int, img_h: int) -> Tuple[float, float]:
    bbox = _region_to_bbox(region, img_w, img_h)
    return _bbox_center(bbox)


def _build_affine(anchors: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]]) -> Optional[np.ndarray]:
    if not anchors:
        return None

    src = np.array([item[0] for item in anchors], dtype=np.float32)
    dst = np.array([item[1] for item in anchors], dtype=np.float32)

    if len(anchors) >= 3:
        matrix, _ = cv2.estimateAffine2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=8.0,
            maxIters=1000,
            confidence=0.99,
        )
        if matrix is None:
            matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=8.0)
        if matrix is not None:
            return matrix.astype(np.float32)

    if len(anchors) == 2:
        sx1, sy1 = src[0]
        sx2, sy2 = src[1]
        dx1, dy1 = dst[0]
        dx2, dy2 = dst[1]
        scale_x = 1.0 if abs(sx2 - sx1) < 1e-6 else (dx2 - dx1) / (sx2 - sx1)
        scale_y = 1.0 if abs(sy2 - sy1) < 1e-6 else (dy2 - dy1) / (sy2 - sy1)
        tx = dx1 - (scale_x * sx1)
        ty = dy1 - (scale_y * sy1)
        return np.array([[scale_x, 0.0, tx], [0.0, scale_y, ty]], dtype=np.float32)

    sx, sy = src[0]
    dx, dy = dst[0]
    return np.array([[1.0, 0.0, dx - sx], [0.0, 1.0, dy - sy]], dtype=np.float32)


def _transform_region(
    region: Dict[str, object],
    affine: Optional[np.ndarray],
    img_w: int,
    img_h: int,
) -> Optional[Dict[str, float]]:
    if affine is None:
        return None

    bbox = _region_to_bbox(region, img_w, img_h)
    x1, y1, x2, y2 = bbox
    pts = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    transformed = cv2.transform(pts, affine)[0]

    xs = np.clip(transformed[:, 0], 0, img_w - 1)
    ys = np.clip(transformed[:, 1], 0, img_h - 1)
    tx1 = int(np.floor(np.min(xs)))
    ty1 = int(np.floor(np.min(ys)))
    tx2 = int(np.ceil(np.max(xs)))
    ty2 = int(np.ceil(np.max(ys)))
    if tx2 <= tx1 or ty2 <= ty1:
        return None

    rows = int(region.get("rows", 1))
    cols = int(region.get("cols", 1))
    return _normalize_region((tx1, ty1, tx2, ty2), rows, cols, img_w, img_h)


def _candidate_count_in_region(
    candidates: Sequence[Candidate],
    region: Dict[str, float],
    img_w: int,
    img_h: int,
) -> int:
    x1, y1, x2, y2 = _region_to_bbox(region, img_w, img_h)
    count = 0
    for cand in candidates:
        cx = float(cand["cx"])
        cy = float(cand["cy"])
        if cx >= x1 and cx <= x2 and cy >= y1 and cy <= y2:
            count += 1
    return count


def _complete_group_section(
    section_name: str,
    template_groups: Sequence[Dict[str, object]],
    detected_regions: Sequence[Optional[Dict[str, float]]],
    labels: Sequence[str],
    rows: int,
    candidates: Sequence[Candidate],
    img_w: int,
    img_h: int,
    affine: Optional[np.ndarray],
    predicted_group_coverage: float,
    predicted_group_min_count: int,
    min_detected_ratio: float,
) -> Tuple[Optional[Dict[str, object]], Dict[str, int], Optional[str]]:
    total = len(template_groups)
    if total == 0:
        return None, {"detected": 0, "predicted": 0, "total": 0}, "template groups are empty"

    groups_out: List[Dict[str, object]] = []
    detected_count = 0
    predicted_count = 0

    for idx, template in enumerate(template_groups):
        cols = int(template.get("cols", 1))
        current = detected_regions[idx] if idx < len(detected_regions) else None
        predicted = False

        if current is None:
            current = _transform_region(template, affine, img_w, img_h)
            predicted = True
        if current is None:
            return (
                None,
                {"detected": detected_count, "predicted": predicted_count, "total": total},
                f"{section_name} group {idx + 1}: missing and affine not available",
            )

        current = dict(current)
        current["rows"] = int(rows)
        current["cols"] = int(cols)

        if "start_question" in template:
            current["start_question"] = int(template["start_question"])
        if "question" in template:
            current["question"] = int(template["question"])

        if predicted:
            expected = max(1, rows * cols)
            min_count = max(int(predicted_group_min_count), int(round(expected * predicted_group_coverage)))
            seen = _candidate_count_in_region(candidates, current, img_w, img_h)
            if seen < min_count:
                return (
                    None,
                    {"detected": detected_count, "predicted": predicted_count, "total": total},
                    f"{section_name} group {idx + 1}: predicted coverage too low ({seen}<{min_count})",
                )
            predicted_count += 1
        else:
            detected_count += 1

        groups_out.append(current)

    ratio = detected_count / max(total, 1)
    if ratio < min_detected_ratio:
        return (
            None,
            {"detected": detected_count, "predicted": predicted_count, "total": total},
            f"{section_name}: detected ratio {ratio:.2f} < {min_detected_ratio:.2f}",
        )

    payload: Dict[str, object] = {
        "groups": groups_out,
        "labels": list(labels),
    }
    if section_name == "dg":
        payload["rows"] = int(rows)
    return payload, {"detected": detected_count, "predicted": predicted_count, "total": total}, None


def _predict_single_region(
    section_name: str,
    template_region: Dict[str, object],
    labels: Sequence[str],
    candidates: Sequence[Candidate],
    img_w: int,
    img_h: int,
    affine: Optional[np.ndarray],
    predicted_coverage: float,
    predicted_min_count: int,
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    predicted = _transform_region(template_region, affine, img_w, img_h)
    if predicted is None:
        return None, f"{section_name}: affine unavailable"

    rows = int(template_region.get("rows", 1))
    cols = int(template_region.get("cols", 1))
    predicted["rows"] = rows
    predicted["cols"] = cols
    predicted["labels"] = list(labels)

    expected = max(1, rows * cols)
    min_count = max(int(predicted_min_count), int(round(expected * predicted_coverage)))
    seen = _candidate_count_in_region(candidates, predicted, img_w, img_h)
    if seen < min_count:
        return None, f"{section_name}: predicted coverage too low ({seen}<{min_count})"
    return predicted, None


def analyze_layout(
    binary: np.ndarray,
    grayscale: Optional[np.ndarray],
    config: Dict[str, object],
) -> Tuple[Optional[Dict[str, object]], List[str], Dict[str, object]]:
    """
    Infer dynamic regions for SBD/MDT/FC/TF/DG from bubble structure.
    """
    warnings: List[str] = []
    diagnostics: Dict[str, object] = {}
    dyn_cfg = config.get("dynamic_layout", {})

    candidates, candidate_stats = detect_bubble_candidates(binary, grayscale=grayscale, config=config)
    diagnostics["candidate_count"] = int(len(candidates))
    diagnostics["candidate_sources"] = candidate_stats

    min_candidates = int(dyn_cfg.get("min_candidate_count", 90))
    if len(candidates) < min_candidates:
        warnings.append("Dynamic layout: not enough bubble candidates, fallback to config.")
        return None, warnings, diagnostics

    h, w = binary.shape[:2]
    diameter = _estimate_diameter(candidates)
    diagnostics["bubble_diameter"] = float(round(diameter, 2))

    neighbor_dx = max(7.0, diameter * float(dyn_cfg.get("component_neighbor_scale_x", 3.2)))
    neighbor_dy = max(7.0, diameter * float(dyn_cfg.get("component_neighbor_scale_y", 3.2)))
    neighbor_dist = max(neighbor_dx, neighbor_dy) * float(dyn_cfg.get("component_neighbor_dist_scale", 1.22))

    component_indices = _connected_components(
        candidates,
        max_dx=neighbor_dx,
        max_dy=neighbor_dy,
        max_dist=neighbor_dist,
    )

    min_points = int(dyn_cfg.get("component_min_points", 8))
    row_cluster_scale = float(dyn_cfg.get("grid_row_tol_scale", 0.88))
    col_cluster_scale = float(dyn_cfg.get("grid_col_tol_scale", 0.88))
    min_occupancy = float(dyn_cfg.get("component_min_occupancy", 0.22))
    min_score = float(dyn_cfg.get("component_min_score", 0.10))

    models: List[Component] = []
    for comp_id, comp in enumerate(component_indices):
        model = _component_to_model(
            comp_id,
            comp,
            candidates,
            diameter=diameter,
            img_w=w,
            img_h=h,
            min_points=min_points,
            row_cluster_scale=row_cluster_scale,
            col_cluster_scale=col_cluster_scale,
        )
        if model is None:
            continue
        if float(model["occupancy"]) < min_occupancy:
            continue
        if float(model["score"]) < min_score:
            continue
        models.append(model)

    models = sorted(models, key=lambda item: (float(item["score"]), float(item["occupancy"]), int(item["count"])), reverse=True)
    diagnostics["component_count"] = int(len(models))
    diagnostics["component_signatures"] = [
        f'{int(comp["rows"])}x{int(comp["cols"])}:{int(comp["count"])}'
        for comp in models[:24]
    ]
    if not models:
        warnings.append("Dynamic layout: no valid grid component found.")
        return None, warnings, diagnostics

    regions_cfg = config["regions"]
    layout_regions: Dict[str, object] = {}
    used_ids: set = set()
    anchors: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    section_status: Dict[str, object] = {}

    # SBD
    sbd_cfg = regions_cfg["sbd"]
    sbd_comp = _pick_single_component(
        models,
        used_ids,
        template_region=sbd_cfg,
        target_rows=int(sbd_cfg["rows"]),
        target_cols=int(sbd_cfg["cols"]),
        row_tol=1,
        col_tol=1,
        img_w=w,
        img_h=h,
    )
    if sbd_comp is not None:
        region = _normalize_region(sbd_comp["bbox"], int(sbd_cfg["rows"]), int(sbd_cfg["cols"]), w, h)
        region["labels"] = list(sbd_cfg["labels"])
        layout_regions["sbd"] = region
        anchors.append((_region_center_px(sbd_cfg, w, h), (float(sbd_comp["cx"]), float(sbd_comp["cy"]))))
        section_status["sbd"] = "detected"
    else:
        warnings.append("Dynamic layout: SBD block not found.")
        section_status["sbd"] = "fallback"

    # MDT
    mdt_cfg = regions_cfg["mdt"]
    mdt_comp = _pick_single_component(
        models,
        used_ids,
        template_region=mdt_cfg,
        target_rows=int(mdt_cfg["rows"]),
        target_cols=int(mdt_cfg["cols"]),
        row_tol=1,
        col_tol=1,
        img_w=w,
        img_h=h,
    )
    if mdt_comp is not None:
        region = _normalize_region(mdt_comp["bbox"], int(mdt_cfg["rows"]), int(mdt_cfg["cols"]), w, h)
        region["labels"] = list(mdt_cfg["labels"])
        layout_regions["mdt"] = region
        anchors.append((_region_center_px(mdt_cfg, w, h), (float(mdt_comp["cx"]), float(mdt_comp["cy"]))))
        section_status["mdt"] = "detected"
    else:
        warnings.append("Dynamic layout: MDT block not found.")
        section_status["mdt"] = "fallback"

    assign_max_score = float(dyn_cfg.get("assign_max_score", 14.0))

    # FC candidates
    fc_cfg = regions_cfg["fc"]
    fc_templates = sorted(fc_cfg["groups"], key=lambda g: int(g["start_question"]))
    fc_assigned = _assign_components_to_groups(
        models,
        used_ids,
        fc_templates,
        target_rows=10,
        target_cols_fn=lambda _: 4,
        row_tol=2,
        col_tol=2,
        img_w=w,
        img_h=h,
        max_score=assign_max_score,
    )
    fc_detected: List[Optional[Dict[str, float]]] = []
    for idx, comp in enumerate(fc_assigned):
        if comp is None:
            fc_detected.append(None)
            continue
        region = _normalize_region(comp["bbox"], 10, 4, w, h)
        region["start_question"] = int(fc_templates[idx]["start_question"])
        fc_detected.append(region)
        anchors.append((_region_center_px(fc_templates[idx], w, h), (float(comp["cx"]), float(comp["cy"]))))

    # TF candidates
    tf_cfg = regions_cfg["tf"]
    tf_templates = sorted(tf_cfg["groups"], key=lambda g: int(g.get("question", 0)))
    tf_assigned = _assign_components_to_groups(
        models,
        used_ids,
        tf_templates,
        target_rows=4,
        target_cols_fn=lambda _: 2,
        row_tol=1,
        col_tol=1,
        img_w=w,
        img_h=h,
        max_score=assign_max_score,
    )
    tf_detected: List[Optional[Dict[str, float]]] = []
    for idx, comp in enumerate(tf_assigned):
        if comp is None:
            tf_detected.append(None)
            continue
        region = _normalize_region(comp["bbox"], 4, 2, w, h)
        region["question"] = int(tf_templates[idx].get("question", idx + 1))
        tf_detected.append(region)
        anchors.append((_region_center_px(tf_templates[idx], w, h), (float(comp["cx"]), float(comp["cy"]))))

    # DG candidates
    dg_cfg = regions_cfg["dg"]
    dg_templates = sorted(dg_cfg["groups"], key=lambda g: int(g.get("question", 0)))
    dg_rows = int(dg_cfg["rows"])
    dg_assigned = _assign_components_to_groups(
        models,
        used_ids,
        dg_templates,
        target_rows=dg_rows,
        target_cols_fn=lambda item: int(item.get("cols", 4)),
        row_tol=3,
        col_tol=3,
        img_w=w,
        img_h=h,
        max_score=assign_max_score + 2.0,
    )
    dg_detected: List[Optional[Dict[str, float]]] = []
    for idx, comp in enumerate(dg_assigned):
        if comp is None:
            dg_detected.append(None)
            continue
        cols = int(dg_templates[idx].get("cols", 4))
        region = _normalize_region(comp["bbox"], dg_rows, cols, w, h)
        region["question"] = int(dg_templates[idx].get("question", idx + 1))
        region["cols"] = cols
        dg_detected.append(region)
        anchors.append((_region_center_px(dg_templates[idx], w, h), (float(comp["cx"]), float(comp["cy"]))))

    affine = _build_affine(anchors)
    diagnostics["anchor_count"] = int(len(anchors))
    diagnostics["affine_ready"] = bool(affine is not None)

    single_pred_cov = float(dyn_cfg.get("single_predicted_coverage", 0.20))
    single_pred_min = int(dyn_cfg.get("single_predicted_min_count", 6))
    if affine is not None and len(anchors) >= 2:
        if "sbd" not in layout_regions:
            predicted_sbd, reason = _predict_single_region(
                "sbd",
                template_region=sbd_cfg,
                labels=sbd_cfg["labels"],
                candidates=candidates,
                img_w=w,
                img_h=h,
                affine=affine,
                predicted_coverage=single_pred_cov,
                predicted_min_count=single_pred_min,
            )
            if predicted_sbd is not None:
                layout_regions["sbd"] = predicted_sbd
                section_status["sbd"] = "predicted"
            else:
                warnings.append(f"Dynamic layout: SBD unresolved ({reason}).")

        if "mdt" not in layout_regions:
            predicted_mdt, reason = _predict_single_region(
                "mdt",
                template_region=mdt_cfg,
                labels=mdt_cfg["labels"],
                candidates=candidates,
                img_w=w,
                img_h=h,
                affine=affine,
                predicted_coverage=single_pred_cov,
                predicted_min_count=max(4, int(single_pred_min * 0.7)),
            )
            if predicted_mdt is not None:
                layout_regions["mdt"] = predicted_mdt
                section_status["mdt"] = "predicted"
            else:
                warnings.append(f"Dynamic layout: MDT unresolved ({reason}).")

    global_pred_coverage = float(dyn_cfg.get("predicted_group_coverage", 0.28))
    fc_pred_coverage = float(dyn_cfg.get("fc_predicted_group_coverage", global_pred_coverage))
    tf_pred_coverage = float(dyn_cfg.get("tf_predicted_group_coverage", max(0.10, global_pred_coverage * 0.5)))
    dg_pred_coverage = float(dyn_cfg.get("dg_predicted_group_coverage", max(0.16, global_pred_coverage * 0.7)))

    fc_payload, fc_stats, fc_reason = _complete_group_section(
        "fc",
        fc_templates,
        fc_detected,
        labels=fc_cfg["labels"],
        rows=10,
        candidates=candidates,
        img_w=w,
        img_h=h,
        affine=affine,
        predicted_group_coverage=fc_pred_coverage,
        predicted_group_min_count=int(dyn_cfg.get("fc_predicted_group_min_count", 6)),
        min_detected_ratio=float(dyn_cfg.get("fc_min_detected_ratio", 0.0)),
    )
    section_status["fc"] = fc_stats
    if fc_payload is not None:
        layout_regions["fc"] = fc_payload
    else:
        warnings.append(f"Dynamic layout: FC unresolved ({fc_reason}).")

    tf_payload, tf_stats, tf_reason = _complete_group_section(
        "tf",
        tf_templates,
        tf_detected,
        labels=tf_cfg["labels"],
        rows=4,
        candidates=candidates,
        img_w=w,
        img_h=h,
        affine=affine,
        predicted_group_coverage=tf_pred_coverage,
        predicted_group_min_count=int(dyn_cfg.get("tf_predicted_group_min_count", 1)),
        min_detected_ratio=float(dyn_cfg.get("tf_min_detected_ratio", 0.0)),
    )
    section_status["tf"] = tf_stats
    if tf_payload is not None:
        layout_regions["tf"] = tf_payload
    else:
        warnings.append(f"Dynamic layout: TF unresolved ({tf_reason}).")

    dg_payload, dg_stats, dg_reason = _complete_group_section(
        "dg",
        dg_templates,
        dg_detected,
        labels=dg_cfg["labels"],
        rows=dg_rows,
        candidates=candidates,
        img_w=w,
        img_h=h,
        affine=affine,
        predicted_group_coverage=dg_pred_coverage,
        predicted_group_min_count=int(dyn_cfg.get("dg_predicted_group_min_count", 7)),
        min_detected_ratio=float(dyn_cfg.get("dg_min_detected_ratio", 0.0)),
    )
    section_status["dg"] = dg_stats
    if dg_payload is not None:
        layout_regions["dg"] = dg_payload
    else:
        warnings.append(f"Dynamic layout: DG unresolved ({dg_reason}).")

    diagnostics["section_status"] = section_status
    diagnostics["resolved_sections"] = sorted(layout_regions.keys())
    if not layout_regions:
        return None, warnings, diagnostics

    min_sections = int(dyn_cfg.get("min_resolved_sections", 4))
    if len(layout_regions) < min_sections:
        warnings.append(
            f"Dynamic layout: resolved {len(layout_regions)} sections (<{min_sections}), fallback to config."
        )
        return None, warnings, diagnostics

    return layout_regions, warnings, diagnostics
