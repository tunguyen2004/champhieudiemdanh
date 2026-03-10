"""Precisely map bubble positions to form regions by examining a filled-in test image."""
import cv2
import numpy as np
import json
from core.preprocessor import load_image, preprocess_to_binary
from core.detector import find_corners, warp_perspective, deskew

config = json.load(open('config.json', 'r', encoding='utf-8'))

# Use a filled-in image for better analysis
img = load_image('anh/test_imga/PhieuQG.0010.jpg')
binary = preprocess_to_binary(img, config)
corners, method = find_corners(img, binary, config)
warped = warp_perspective(img, corners, 1800, 2500)
warped, _ = deskew(warped)
warped = cv2.resize(warped, (1800, 2500))

gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
h, w = thresh.shape

# Find all bubbles (small dark regions)
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

print(f"Total bubbles found: {len(bubbles)}")

# Separate bubbles into regions based on the form structure
# From the horizontal line analysis:
#   y=713 (0.285) = divider between header and FC/TF
#   y=1247 (0.499) = divider between FC/TF and lower section
#   y=1657 (0.663) = divider 
#   y=1751 (0.700) = divider before DG

# Header bubbles (SBD, MDT) - y < 0.285
header_bubbles = [(cx, cy, bw, bh, a) for cx, cy, bw, bh, a in bubbles if cy < h * 0.285]

# FC/TF area bubbles - 0.285 < y < 0.50
fc_tf_bubbles = [(cx, cy, bw, bh, a) for cx, cy, bw, bh, a in bubbles if h * 0.285 < cy < h * 0.50]

# Mid section - 0.50 < y < 0.70
mid_bubbles = [(cx, cy, bw, bh, a) for cx, cy, bw, bh, a in bubbles if h * 0.50 < cy < h * 0.70]

# DG area - y > 0.70
dg_bubbles = [(cx, cy, bw, bh, a) for cx, cy, bw, bh, a in bubbles if cy > h * 0.70]

print(f"\nHeader bubbles: {len(header_bubbles)}")
print(f"FC/TF bubbles: {len(fc_tf_bubbles)}")
print(f"Mid bubbles: {len(mid_bubbles)}")
print(f"DG bubbles: {len(dg_bubbles)}")

# === Analyze header bubbles for SBD/MDT ===
print("\n=== HEADER REGION (SBD/MDT) ===")
if header_bubbles:
    hb_x = [b[0] for b in header_bubbles]
    hb_y = [b[1] for b in header_bubbles]
    print(f"X range: {min(hb_x)}-{max(hb_x)} ({min(hb_x)/w:.3f}-{max(hb_x)/w:.3f})")
    print(f"Y range: {min(hb_y)}-{max(hb_y)} ({min(hb_y)/h:.3f}-{max(hb_y)/h:.3f})")
    
    # Cluster by X to find columns in header
    hb_sorted = sorted(header_bubbles, key=lambda b: b[0])
    x_cols = []
    current = [hb_sorted[0][0]]
    for b in hb_sorted[1:]:
        if b[0] - current[-1] < 20:
            current.append(b[0])
        else:
            x_cols.append((int(np.mean(current)), len(current)))
            current = [b[0]]
    x_cols.append((int(np.mean(current)), len(current)))
    
    print("Header bubble columns:")
    for x, cnt in x_cols:
        if cnt >= 3:
            print(f"  x={x:4d} ({x/w:.3f})  count={cnt}")

# === Analyze FC/TF area ===  
print("\n=== FC/TF REGION ===")
if fc_tf_bubbles:
    ft_x = [b[0] for b in fc_tf_bubbles]
    ft_y = [b[1] for b in fc_tf_bubbles]
    print(f"X range: {min(ft_x)}-{max(ft_x)} ({min(ft_x)/w:.3f}-{max(ft_x)/w:.3f})")
    print(f"Y range: {min(ft_y)}-{max(ft_y)} ({min(ft_y)/h:.3f}-{max(ft_y)/h:.3f})")
    
    # Split by vertical line at x~880 (0.489) 
    left_ft = [b for b in fc_tf_bubbles if b[0] < w * 0.489]
    right_ft = [b for b in fc_tf_bubbles if b[0] >= w * 0.489]
    
    print(f"\nLeft half: {len(left_ft)} bubbles")
    if left_ft:
        # Find columns
        left_sorted = sorted(left_ft, key=lambda b: b[0])
        x_cols = []
        current = [left_sorted[0][0]]
        for b in left_sorted[1:]:
            if b[0] - current[-1] < 20:
                current.append(b[0])
            else:
                x_cols.append((int(np.mean(current)), len(current)))
                current = [b[0]]
        x_cols.append((int(np.mean(current)), len(current)))
        print("  Left columns:")
        for x, cnt in x_cols:
            if cnt >= 3:
                print(f"    x={x:4d} ({x/w:.3f})  count={cnt}")
    
    print(f"\nRight half: {len(right_ft)} bubbles")
    if right_ft:
        right_sorted = sorted(right_ft, key=lambda b: b[0])
        x_cols = []
        current = [right_sorted[0][0]]
        for b in right_sorted[1:]:
            if b[0] - current[-1] < 20:
                current.append(b[0])
            else:
                x_cols.append((int(np.mean(current)), len(current)))
                current = [b[0]]
        x_cols.append((int(np.mean(current)), len(current)))
        print("  Right columns:")
        for x, cnt in x_cols:
            if cnt >= 3:
                print(f"    x={x:4d} ({x/w:.3f})  count={cnt}")

    # Find rows in entire FC/TF area
    ft_sorted_y = sorted(fc_tf_bubbles, key=lambda b: b[1])
    y_rows = []
    current = [ft_sorted_y[0][1]]
    for b in ft_sorted_y[1:]:
        if b[1] - current[-1] < 15:
            current.append(b[1])
        else:
            y_rows.append((int(np.mean(current)), len(current)))
            current = [b[1]]
    y_rows.append((int(np.mean(current)), len(current)))
    
    print(f"\nFC/TF bubble rows:")
    for y, cnt in y_rows:
        print(f"  y={y:4d} ({y/h:.3f})  count={cnt}")

# === Analyze mid section (likely TF continuation or transition) ===
print("\n=== MID REGION ===")
if mid_bubbles:
    mid_x = [b[0] for b in mid_bubbles]
    mid_y = [b[1] for b in mid_bubbles]
    print(f"X range: {min(mid_x)}-{max(mid_x)} ({min(mid_x)/w:.3f}-{max(mid_x)/w:.3f})")
    print(f"Y range: {min(mid_y)}-{max(mid_y)} ({min(mid_y)/h:.3f}-{max(mid_y)/h:.3f})")
    
    # Find rows
    mid_sorted_y = sorted(mid_bubbles, key=lambda b: b[1])
    y_rows = []
    current = [mid_sorted_y[0][1]]
    for b in mid_sorted_y[1:]:
        if b[1] - current[-1] < 15:
            current.append(b[1])
        else:
            y_rows.append((int(np.mean(current)), len(current)))
            current = [b[1]]
    y_rows.append((int(np.mean(current)), len(current)))
    
    print("Mid bubble rows:")
    for y, cnt in y_rows:
        print(f"  y={y:4d} ({y/h:.3f})  count={cnt}")
    
    # Find columns
    mid_sorted_x = sorted(mid_bubbles, key=lambda b: b[0])
    x_cols = []
    current = [mid_sorted_x[0][0]]
    for b in mid_sorted_x[1:]:
        if b[0] - current[-1] < 20:
            current.append(b[0])
        else:
            x_cols.append((int(np.mean(current)), len(current)))
            current = [b[0]]
    x_cols.append((int(np.mean(current)), len(current)))
    print("Mid bubble columns:")
    for x, cnt in x_cols:
        if cnt >= 3:
            print(f"  x={x:4d} ({x/w:.3f})  count={cnt}")

# === Analyze DG area ===
print("\n=== DG REGION ===")
if dg_bubbles:
    dg_x = [b[0] for b in dg_bubbles]
    dg_y = [b[1] for b in dg_bubbles]
    print(f"X range: {min(dg_x)}-{max(dg_x)} ({min(dg_x)/w:.3f}-{max(dg_x)/w:.3f})")
    print(f"Y range: {min(dg_y)}-{max(dg_y)} ({min(dg_y)/h:.3f}-{max(dg_y)/h:.3f})")
    
    # Find rows
    dg_sorted_y = sorted(dg_bubbles, key=lambda b: b[1])
    y_rows = []
    current = [dg_sorted_y[0][1]]
    for b in dg_sorted_y[1:]:
        if b[1] - current[-1] < 15:
            current.append(b[1])
        else:
            y_rows.append((int(np.mean(current)), len(current)))
            current = [b[1]]
    y_rows.append((int(np.mean(current)), len(current)))
    
    print("DG bubble rows:")
    for y, cnt in y_rows:
        print(f"  y={y:4d} ({y/h:.3f})  count={cnt}")
