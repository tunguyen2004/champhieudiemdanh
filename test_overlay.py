"""Detailed analysis: Compare expected vs actual bubble positions on an aligned image."""
import cv2
import numpy as np
from core.preprocessor import load_image, preprocess, preprocess_to_binary
from core.detector import find_corners, warp_perspective, deskew, align_to_template
import json

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# Test with PhieuQG.0010 (known marks)
img = load_image("anh/test_imga/PhieuQG.0010.jpg")
gray = preprocess(img, config)
binary = preprocess_to_binary(img, config)
corners, method = find_corners(img, binary, config)

W, H = 1800, 2500
warped = warp_perspective(img, corners, W, H)
warped, skew = deskew(warped)
if abs(skew) > 0.3:
    warped = cv2.resize(warped, (W, H), interpolation=cv2.INTER_CUBIC)
aligned, ok = align_to_template(warped, config)

print(f"Aligned: {ok}")

# Save aligned image with grid overlay for all regions
vis = aligned.copy()
regions = config["regions"]

# Draw SBD region
sbd = regions["sbd"]
x1, y1 = int(sbd["x1"]*W), int(sbd["y1"]*H)
x2, y2 = int(sbd["x2"]*W), int(sbd["y2"]*H)
cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(vis, "SBD", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Draw individual SBD cells
for col in range(sbd["cols"]):
    for row in range(sbd["rows"]):
        cx1 = x1 + int(col * (x2 - x1) / sbd["cols"])
        cy1 = y1 + int(row * (y2 - y1) / sbd["rows"])
        cx2 = x1 + int((col + 1) * (x2 - x1) / sbd["cols"])
        cy2 = y1 + int((row + 1) * (y2 - y1) / sbd["rows"])
        cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 200, 0), 1)

# Draw MDT region
mdt = regions["mdt"]
x1, y1 = int(mdt["x1"]*W), int(mdt["y1"]*H)
x2, y2 = int(mdt["x2"]*W), int(mdt["y2"]*H)
cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
cv2.putText(vis, "MDT", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Draw FC regions
for i, grp in enumerate(regions["fc"]["groups"]):
    x1, y1 = int(grp["x1"]*W), int(grp["y1"]*H)
    x2, y2 = int(grp["x2"]*W), int(grp["y2"]*H)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(vis, f"FC{i+1}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # Draw individual cells
    for row in range(grp["rows"]):
        for col in range(grp["cols"]):
            cx1 = x1 + int(col * (x2 - x1) / grp["cols"])
            cy1 = y1 + int(row * (y2 - y1) / grp["rows"])
            cx2 = x1 + int((col + 1) * (x2 - x1) / grp["cols"])
            cy2 = y1 + int((row + 1) * (y2 - y1) / grp["rows"])
            cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 0, 200), 1)

# Draw TF regions
for i, grp in enumerate(regions["tf"]["groups"]):
    x1, y1 = int(grp["x1"]*W), int(grp["y1"]*H)
    x2, y2 = int(grp["x2"]*W), int(grp["y2"]*H)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)

# Draw DG regions
for grp in regions["dg"]["groups"]:
    x1, y1 = int(grp["x1"]*W), int(grp["y1"]*H)
    x2, y2 = int(grp["x2"]*W), int(grp["y2"]*H)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(vis, f"DG{grp['question']}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

cv2.imwrite("anh/aligned_grid_overlay.jpg", vis)
print("Saved anh/aligned_grid_overlay.jpg")

# Also check: what are fill ratios at specific SBD positions?
aligned_binary = preprocess_to_binary(aligned, config)

print("\n=== SBD fill ratios ===")
sbd_cfg = regions["sbd"]
sx1, sy1 = int(sbd_cfg["x1"]*W), int(sbd_cfg["y1"]*H)
sx2, sy2 = int(sbd_cfg["x2"]*W), int(sbd_cfg["y2"]*H)
cell_w = (sx2 - sx1) / sbd_cfg["cols"]
cell_h = (sy2 - sy1) / sbd_cfg["rows"]
margin = config.get("bubble_margin", 0.15)

for col in range(sbd_cfg["cols"]):
    ratios = []
    for row in range(sbd_cfg["rows"]):
        cx1 = sx1 + int(col * cell_w + cell_w * margin)
        cy1 = sy1 + int(row * cell_h + cell_h * margin)
        cx2 = sx1 + int((col + 1) * cell_w - cell_w * margin)
        cy2 = sy1 + int((row + 1) * cell_h - cell_h * margin)
        roi = aligned_binary[cy1:cy2, cx1:cx2]
        if roi.size > 0:
            ratio = cv2.countNonZero(roi) / roi.size
        else:
            ratio = 0
        ratios.append(ratio)
    print(f"  Col {col}: {[f'{r:.2f}' for r in ratios]}")
    max_idx = np.argmax(ratios)
    print(f"    -> digit {max_idx} (ratio={ratios[max_idx]:.2f})")
