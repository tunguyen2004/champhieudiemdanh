"""Analyze exact column positions within rows to determine group structure."""
import cv2
import numpy as np
import json
from core.preprocessor import load_image, preprocess_to_binary
from core.detector import find_corners, warp_perspective, deskew

config = json.load(open('config.json', 'r', encoding='utf-8'))

# Use template for cleaner analysis
img = load_image('anh/sample_image/Copy of PhieuTracNghiepTHPT2025.png')
binary = preprocess_to_binary(img, config)
corners, method = find_corners(img, binary, config)
warped = warp_perspective(img, corners, 1800, 2500)
warped, _ = deskew(warped)
warped = cv2.resize(warped, (1800, 2500))
gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
h, w = thresh.shape

# Find all bubbles
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
bubbles = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 200 < area < 4000:
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        if 0.4 < aspect < 2.5 and bw > 8 and bh > 8:
            cx, cy = x + bw // 2, y + bh // 2
            bubbles.append((cx, cy, bw, bh, area))

# Group bubbles by Y (into rows)
bubbles.sort(key=lambda b: b[1])
rows = {}
current_key = bubbles[0][1]
current_group = [bubbles[0]]
for b in bubbles[1:]:
    if b[1] - current_group[-1][1] < 15:
        current_group.append(b)
    else:
        avg_y = int(np.mean([bb[1] for bb in current_group]))
        rows[avg_y] = current_group
        current_group = [b]
        current_key = b[1]
avg_y = int(np.mean([bb[1] for bb in current_group]))
rows[avg_y] = current_group

# Analyze specific rows
print("=== HEADER REGION: SBD/MDT columns ===")
# Look at rows in header area (y < 700) with right-side columns
for y_key in sorted(rows.keys()):
    if y_key > 700:
        break
    row_bubbles = rows[y_key]
    # Focus on right side (x > 1200) where SBD/MDT are
    right_bubbles = [b for b in row_bubbles if b[0] > 1200]
    if len(right_bubbles) >= 3:
        x_positions = sorted([b[0] for b in right_bubbles])
        print(f"  y={y_key:4d} ({y_key/h:.3f}): {len(right_bubbles)} bubbles, x={x_positions}")

print("\n=== FC ROW ANALYSIS ===")
# Analyze rows in FC area (y: 0.35 - 0.53)
fc_rows = {k: v for k, v in rows.items() if 0.35*h <= k <= 0.53*h and len(v) >= 15}
for y_key in sorted(fc_rows.keys())[:3]:
    row_bubbles = fc_rows[y_key]
    x_positions = sorted([b[0] for b in row_bubbles])
    print(f"  y={y_key:4d} ({y_key/h:.3f}): {len(row_bubbles)} bubbles")
    print(f"    x positions: {x_positions}")
    
    # Find gaps between columns to identify groups
    gaps = []
    for i in range(1, len(x_positions)):
        gap = x_positions[i] - x_positions[i-1]
        if gap > 50:
            gaps.append((x_positions[i-1], x_positions[i], gap))
    if gaps:
        print(f"    Major gaps (>50px): {[(f'{a}->{b} gap={g}') for a,b,g in gaps]}")

print("\n=== TF ROW ANALYSIS ===")
# Analyze rows in TF area (y: 0.55 - 0.67)
tf_rows = {k: v for k, v in rows.items() if 0.55*h <= k <= 0.67*h and len(v) >= 10}
for y_key in sorted(tf_rows.keys()):
    row_bubbles = tf_rows[y_key]
    x_positions = sorted([b[0] for b in row_bubbles])
    print(f"  y={y_key:4d} ({y_key/h:.3f}): {len(row_bubbles)} bubbles")
    print(f"    x positions: {x_positions}")
    
    gaps = []
    for i in range(1, len(x_positions)):
        gap = x_positions[i] - x_positions[i-1]
        if gap > 50:
            gaps.append((x_positions[i-1], x_positions[i], gap))
    if gaps:
        print(f"    Major gaps: {[(f'{a}->{b} gap={g}') for a,b,g in gaps]}")

print("\n=== DG ROW ANALYSIS ===")
# Analyze first data row in DG area (y > 0.79)
dg_rows = {k: v for k, v in rows.items() if k > 0.79*h and len(v) >= 15}
first_two = sorted(dg_rows.keys())[:2]
for y_key in first_two:
    row_bubbles = dg_rows[y_key]
    x_positions = sorted([b[0] for b in row_bubbles])
    print(f"  y={y_key:4d} ({y_key/h:.3f}): {len(row_bubbles)} bubbles")
    print(f"    x positions: {x_positions}")
    
    gaps = []
    for i in range(1, len(x_positions)):
        gap = x_positions[i] - x_positions[i-1]
        if gap > 50:
            gaps.append((x_positions[i-1], x_positions[i], gap))
    if gaps:
        print(f"    Major gaps: {[(f'{a}->{b} gap={g}') for a,b,g in gaps]}")

# Summary: identify all FC bubble rows with their Y positions
print("\n=== ALL BUBBLE ROW SUMMARY ===")
for y_key in sorted(rows.keys()):
    cnt = len(rows[y_key])
    if cnt >= 15:
        region = "??"
        if y_key < h*0.285:
            region = "HDR"
        elif y_key < h*0.53:
            region = "FC "
        elif y_key < h*0.55:
            region = "DIV"
        elif y_key < h*0.67:
            region = "TF "
        elif y_key < h*0.72:
            region = "DIV"
        else:
            region = "DG "
        pct = y_key/h
        print(f"  [{region}] y={y_key:4d} ({pct:.3f}): {cnt:3d} bubbles")
