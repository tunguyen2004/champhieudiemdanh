"""
Dynamic layout detection for the THPT answer sheet.

This module keeps the sheet schema (row/column counts) but replaces fixed
coordinate regions with regions detected from the warped sheet structure.
"""
import copy

import cv2
import numpy as np


def _to_binary(gray):
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _line_masks(binary):
    h, w = binary.shape[:2]
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 40, 25), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 60, 25)))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    lines = cv2.bitwise_or(h_lines, v_lines)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    return h_lines, v_lines, closed


def _candidate_rects(line_mask):
    h, w = line_mask.shape[:2]
    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < w * h * 0.001 or bw < 40 or bh < 40:
            continue
        rects.append({
            "x": x,
            "y": y,
            "w": bw,
            "h": bh,
            "area": area,
            "aspect": bw / max(bh, 1),
            "cx": x + bw / 2,
            "cy": y + bh / 2,
        })
    return rects


def _cluster_positions(values, min_gap=8):
    if len(values) == 0:
        return []
    values = sorted(int(v) for v in values)
    clusters = []
    for value in values:
        if not clusters or value - clusters[-1][-1] > min_gap:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [int(np.mean(cluster)) for cluster in clusters]


def _project_line_positions(mask, rect, axis, coverage_threshold):
    x, y, w, h = rect
    roi = mask[y:y + h, x:x + w]
    if roi.size == 0:
        return []
    if axis == "x":
        projection = np.sum(roi > 0, axis=0)
        peaks = np.where(projection > h * coverage_threshold)[0]
        return [x + pos for pos in _cluster_positions(peaks)]
    projection = np.sum(roi > 0, axis=1)
    peaks = np.where(projection > w * coverage_threshold)[0]
    return [y + pos for pos in _cluster_positions(peaks)]


def _norm_rect(rect, img_w, img_h):
    x, y, w, h = rect
    return {
        "x1": max(x / img_w, 0.0),
        "y1": max(y / img_h, 0.0),
        "x2": min((x + w) / img_w, 1.0),
        "y2": min((y + h) / img_h, 1.0),
    }


def _apply_id_regions(dynamic_config, rects, img_w, img_h):
    id_candidates = [
        rect for rect in rects
        if rect["cy"] < img_h * 0.35
        and rect["x"] > img_w * 0.55
        and rect["h"] > img_h * 0.12
        and 0.15 <= rect["aspect"] <= 0.65
    ]
    if len(id_candidates) < 2:
        return False

    id_candidates = sorted(id_candidates, key=lambda r: r["x"])[:2]
    sbd_rect, mdt_rect = id_candidates[0], id_candidates[1]

    sbd = dynamic_config["regions"]["sbd"]
    mdt = dynamic_config["regions"]["mdt"]
    sbd.update(_norm_rect((sbd_rect["x"], sbd_rect["y"], sbd_rect["w"], sbd_rect["h"]), img_w, img_h))
    mdt.update(_norm_rect((mdt_rect["x"], mdt_rect["y"], mdt_rect["w"], mdt_rect["h"]), img_w, img_h))
    return True


def _apply_fc_regions(dynamic_config, rects, img_w, img_h):
    fc_candidates = [
        rect for rect in rects
        if img_h * 0.28 < rect["cy"] < img_h * 0.58
        and img_w * 0.12 < rect["w"] < img_w * 0.30
        and img_h * 0.14 < rect["h"] < img_h * 0.26
        and 0.45 < rect["aspect"] < 1.05
    ]
    if len(fc_candidates) < 4:
        return False

    fc_candidates = sorted(fc_candidates, key=lambda r: r["x"])[:4]
    starts = [1, 11, 21, 31]
    groups = []
    for rect, start in zip(fc_candidates, starts):
        x = rect["x"] + rect["w"] * 0.13
        y = rect["y"] + rect["h"] * 0.09
        w = rect["w"] * 0.84
        h = rect["h"] * 0.88
        group = _norm_rect((x, y, w, h), img_w, img_h)
        group.update({"rows": 10, "cols": 4, "start_question": start})
        groups.append(group)
    dynamic_config["regions"]["fc"]["groups"] = groups
    return True


def _apply_tf_regions(dynamic_config, rects, img_w, img_h):
    tf_candidates = [
        rect for rect in rects
        if img_h * 0.52 < rect["cy"] < img_h * 0.75
        and img_w * 0.12 < rect["w"] < img_w * 0.30
        and img_h * 0.08 < rect["h"] < img_h * 0.16
        and 0.9 < rect["aspect"] < 1.6
    ]
    if len(tf_candidates) < 4:
        return False

    tf_candidates = sorted(tf_candidates, key=lambda r: r["x"])[:4]
    groups = []
    question = 1
    for rect in tf_candidates:
        y = rect["y"] + rect["h"] * 0.40
        h = rect["h"] * 0.58
        left_x = rect["x"] + rect["w"] * 0.10
        split_x = rect["x"] + rect["w"] * 0.535
        right_x2 = rect["x"] + rect["w"] * 0.95
        for x1, x2 in ((left_x, split_x), (split_x, right_x2)):
            group = _norm_rect((x1, y, x2 - x1, h), img_w, img_h)
            group.update({"question": question, "rows": 4, "cols": 2})
            groups.append(group)
            question += 1
    dynamic_config["regions"]["tf"]["groups"] = groups
    return True


def _apply_dg_regions(dynamic_config, rects, v_lines, h_lines, img_w, img_h):
    dg_candidates = [
        rect for rect in rects
        if rect["cy"] > img_h * 0.68
        and rect["w"] > img_w * 0.60
        and rect["h"] > img_h * 0.18
        and rect["aspect"] > 1.5
    ]
    if not dg_candidates:
        return False

    rect = max(dg_candidates, key=lambda r: r["area"])
    box = (rect["x"], rect["y"], rect["w"], rect["h"])
    x_lines = _project_line_positions(v_lines, box, "x", 0.35)
    y_lines = _project_line_positions(h_lines, box, "y", 0.10)
    if len(x_lines) < 7:
        return False

    x_lines = x_lines[:7]
    y_start = y_lines[2] if len(y_lines) >= 3 else rect["y"] + int(rect["h"] * 0.18)
    y_end = y_lines[-1] if len(y_lines) >= 4 else rect["y"] + int(rect["h"] * 0.98)
    if y_end <= y_start:
        return False

    groups = []
    for idx in range(6):
        x1 = x_lines[idx]
        x2 = x_lines[idx + 1]
        pad = max(int((x2 - x1) * 0.02), 1)
        group = _norm_rect((x1 + pad, y_start, (x2 - x1) - 2 * pad, y_end - y_start), img_w, img_h)
        group.update({"question": idx + 1, "cols": 4})
        groups.append(group)

    dynamic_config["regions"]["dg"]["groups"] = groups
    return True


def apply_dynamic_layout(warped_gray, base_config):
    """Return a config copy with detected regions plus a short layout status list."""
    dynamic_config = copy.deepcopy(base_config)
    binary = _to_binary(warped_gray)
    h_lines, v_lines, line_mask = _line_masks(binary)
    rects = _candidate_rects(line_mask)
    h, w = binary.shape[:2]

    status = []
    if _apply_id_regions(dynamic_config, rects, w, h):
        status.append("id")
    if _apply_fc_regions(dynamic_config, rects, w, h):
        status.append("fc")
    if _apply_tf_regions(dynamic_config, rects, w, h):
        status.append("tf")
    if _apply_dg_regions(dynamic_config, rects, v_lines, h_lines, w, h):
        status.append("dg")

    dynamic_config["_dynamic_layout"] = status
    return dynamic_config
