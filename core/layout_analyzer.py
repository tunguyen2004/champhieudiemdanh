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


def _contour_circularity(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return 0.0
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0:
        return 0.0
    return float((4.0 * np.pi * area) / (perimeter * perimeter))


def detect_bubble_candidates(
    binary: np.ndarray,
    config: Optional[Dict[str, object]] = None
) -> List[Candidate]:
    """
    Detect candidate bubble centers from a binary image.
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
    candidates: List[Candidate] = []

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

        cx = x + (cw / 2.0)
        cy = y + (ch / 2.0)
        candidates.append({
            "cx": float(cx),
            "cy": float(cy),
            "x": float(x),
            "y": float(y),
            "w": float(cw),
            "h": float(ch),
            "area": area,
            "circularity": circularity,
        })

    return candidates


def _estimate_diameter(candidates: Sequence[Candidate]) -> float:
    if not candidates:
        return 8.0
    areas = [max(float(item["area"]), 1.0) for item in candidates]
    diameters = [np.sqrt((4.0 * area) / np.pi) for area in areas]
    return float(np.median(diameters))


def _connected_components(
    candidates: Sequence[Candidate],
    max_dx: float,
    max_dy: float
) -> List[List[int]]:
    n = len(candidates)
    visited = [False] * n
    components: List[List[int]] = []

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
                if abs(c1["cx"] - c2["cx"]) <= max_dx and abs(c1["cy"] - c2["cy"]) <= max_dy:
                    visited[j] = True
                    stack.append(j)
                    group.append(j)

        components.append(group)

    return components


def _component_to_model(
    component_id: int,
    indices: Sequence[int],
    candidates: Sequence[Candidate],
    diameter: float,
    img_w: int,
    img_h: int
) -> Optional[Component]:
    if len(indices) < 10:
        return None

    points = [(float(candidates[idx]["cx"]), float(candidates[idx]["cy"])) for idx in indices]
    grid = infer_grid_signature(
        points,
        row_tolerance=max(3.0, diameter * 0.70),
        col_tolerance=max(3.0, diameter * 0.70),
    )
    rows = int(grid["rows"])
    cols = int(grid["cols"])
    if rows < 2 or cols < 2:
        return None

    bbox = bbox_from_points(
        points,
        pad_x=max(2.0, diameter * 1.1),
        pad_y=max(2.0, diameter * 1.1),
        img_w=img_w,
        img_h=img_h,
    )

    return {
        "id": component_id,
        "rows": rows,
        "cols": cols,
        "count": len(indices),
        "occupancy": float(grid["occupancy"]),
        "score": float(grid["score"]),
        "bbox": bbox,
        "points": points,
    }


def _component_match_score(component: Component, target_rows: int, target_cols: int) -> float:
    row_diff = abs(int(component["rows"]) - target_rows)
    col_diff = abs(int(component["cols"]) - target_cols)
    density_penalty = max(0.0, 0.80 - float(component["occupancy"])) * 2.0
    return row_diff * 3.0 + col_diff * 3.0 + density_penalty - float(component["score"])


def _pick_components(
    components: Sequence[Component],
    used_ids: set,
    target_rows: int,
    target_cols: int,
    count: int
) -> List[Component]:
    candidates = [
        comp for comp in components
        if comp["id"] not in used_ids
        and abs(int(comp["rows"]) - target_rows) <= 1
        and abs(int(comp["cols"]) - target_cols) <= 1
    ]
    ranked = sorted(
        candidates,
        key=lambda comp: _component_match_score(comp, target_rows, target_cols)
    )
    picked = ranked[:count]
    for item in picked:
        used_ids.add(item["id"])
    return picked


def _normalize_region(
    bbox: Tuple[int, int, int, int],
    rows: int,
    cols: int,
    img_w: int,
    img_h: int
) -> Dict[str, float]:
    x1, y1, x2, y2 = bbox
    return {
        "x1": max(0.0, x1 / max(img_w, 1)),
        "y1": max(0.0, y1 / max(img_h, 1)),
        "x2": min(1.0, x2 / max(img_w, 1)),
        "y2": min(1.0, y2 / max(img_h, 1)),
        "rows": int(rows),
        "cols": int(cols),
    }


def analyze_layout(
    binary: np.ndarray,
    config: Dict[str, object]
) -> Tuple[Optional[Dict[str, object]], List[str], Dict[str, object]]:
    """
    Infer dynamic regions for SBD/MDT/FC/TF/DG based on bubble structure.
    """
    warnings: List[str] = []
    diagnostics: Dict[str, object] = {}

    candidates = detect_bubble_candidates(binary, config=config)
    diagnostics["candidate_count"] = len(candidates)
    if len(candidates) < 120:
        warnings.append("Dynamic layout: khong du bubble candidate, fallback ve config.")
        return None, warnings, diagnostics

    h, w = binary.shape[:2]
    diameter = _estimate_diameter(candidates)
    diagnostics["bubble_diameter"] = diameter

    components_idx = _connected_components(
        candidates,
        max_dx=max(8.0, diameter * 4.5),
        max_dy=max(8.0, diameter * 4.5),
    )
    models: List[Component] = []
    for comp_id, comp in enumerate(components_idx):
        model = _component_to_model(comp_id, comp, candidates, diameter, w, h)
        if model is not None:
            models.append(model)

    diagnostics["component_count"] = len(models)
    if not models:
        warnings.append("Dynamic layout: khong suy ra duoc component luoi hop le.")
        return None, warnings, diagnostics

    regions_cfg = config["regions"]
    used_ids: set = set()
    layout_regions: Dict[str, object] = {}

    # SBD
    sbd_target = regions_cfg["sbd"]
    picked = _pick_components(models, used_ids, int(sbd_target["rows"]), int(sbd_target["cols"]), 1)
    if picked:
        region = _normalize_region(picked[0]["bbox"], int(sbd_target["rows"]), int(sbd_target["cols"]), w, h)
        region["labels"] = list(sbd_target["labels"])
        layout_regions["sbd"] = region
    else:
        warnings.append("Dynamic layout: khong tim thay block SBD, giu config cu.")

    # MDT
    mdt_target = regions_cfg["mdt"]
    picked = _pick_components(models, used_ids, int(mdt_target["rows"]), int(mdt_target["cols"]), 1)
    if picked:
        region = _normalize_region(picked[0]["bbox"], int(mdt_target["rows"]), int(mdt_target["cols"]), w, h)
        region["labels"] = list(mdt_target["labels"])
        layout_regions["mdt"] = region
    else:
        warnings.append("Dynamic layout: khong tim thay block MDT, giu config cu.")

    # FC groups
    fc_template_groups = regions_cfg["fc"]["groups"]
    fc_count = len(fc_template_groups)
    fc_picked = _pick_components(models, used_ids, 10, 4, fc_count)
    if len(fc_picked) == fc_count:
        fc_picked = sorted(fc_picked, key=lambda item: item["bbox"][0])
        fc_groups: List[Dict[str, object]] = []
        template_order = sorted(fc_template_groups, key=lambda g: g["start_question"])
        for comp, template in zip(fc_picked, template_order):
            group = _normalize_region(comp["bbox"], 10, 4, w, h)
            group["start_question"] = int(template["start_question"])
            fc_groups.append(group)
        layout_regions["fc"] = {
            "groups": fc_groups,
            "labels": list(regions_cfg["fc"]["labels"]),
        }
    else:
        warnings.append("Dynamic layout: so nhom FC khong du, giu config cu.")

    # TF groups
    tf_template_groups = regions_cfg["tf"]["groups"]
    tf_count = len(tf_template_groups)
    tf_picked = _pick_components(models, used_ids, 4, 2, tf_count)
    if len(tf_picked) == tf_count:
        tf_picked = sorted(tf_picked, key=lambda item: item["bbox"][0])
        tf_groups: List[Dict[str, object]] = []
        template_order = sorted(tf_template_groups, key=lambda g: int(g.get("question", 0)))
        for comp, template in zip(tf_picked, template_order):
            group = _normalize_region(comp["bbox"], 4, 2, w, h)
            group["question"] = int(template.get("question", len(tf_groups) + 1))
            tf_groups.append(group)
        layout_regions["tf"] = {
            "groups": tf_groups,
            "labels": list(regions_cfg["tf"]["labels"]),
        }
    else:
        warnings.append("Dynamic layout: so nhom TF khong du, giu config cu.")

    # DG groups
    dg_template_groups = regions_cfg["dg"]["groups"]
    dg_rows = int(regions_cfg["dg"]["rows"])
    dg_count = len(dg_template_groups)
    dg_cols = int(dg_template_groups[0]["cols"]) if dg_template_groups else 4
    dg_picked = _pick_components(models, used_ids, dg_rows, dg_cols, dg_count)
    if len(dg_picked) == dg_count:
        dg_picked = sorted(dg_picked, key=lambda item: item["bbox"][0])
        dg_groups: List[Dict[str, object]] = []
        template_order = sorted(dg_template_groups, key=lambda g: int(g.get("question", 0)))
        for comp, template in zip(dg_picked, template_order):
            group = _normalize_region(comp["bbox"], dg_rows, int(template["cols"]), w, h)
            group["question"] = int(template["question"])
            group["cols"] = int(template["cols"])
            dg_groups.append(group)
        layout_regions["dg"] = {
            "groups": dg_groups,
            "rows": dg_rows,
            "labels": list(regions_cfg["dg"]["labels"]),
        }
    else:
        warnings.append("Dynamic layout: so nhom DG khong du, giu config cu.")

    if not layout_regions:
        return None, warnings, diagnostics

    dyn_cfg = config.get("dynamic_layout", {})
    min_sections = int(dyn_cfg.get("min_resolved_sections", 4))
    if len(layout_regions) < min_sections:
        warnings.append(
            f"Dynamic layout: chi resolve duoc {len(layout_regions)} section (<{min_sections}), fallback ve config."
        )
        return None, warnings, diagnostics

    diagnostics["resolved_sections"] = sorted(layout_regions.keys())
    return layout_regions, warnings, diagnostics
