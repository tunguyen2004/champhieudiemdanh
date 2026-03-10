"""Get complete coordinates for config.json calibration using ink detection."""
import cv2
import numpy as np
import json
from core.preprocessor import load_image, preprocess_to_binary
from core.detector import find_sheet_by_ink, warp_perspective, deskew

config = json.load(open('config.json', 'r', encoding='utf-8'))

# Use filled test image with INK detection (our primary method)
img = load_image('anh/test_imga/PhieuQG.0010.jpg')
corners = find_sheet_by_ink(img)
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

# Cluster into rows
bubbles.sort(key=lambda b: b[1])
rows = []
current_group = [bubbles[0]]
for b in bubbles[1:]:
    if b[1] - current_group[-1][1] < 15:
        current_group.append(b)
    else:
        avg_y = int(np.mean([bb[1] for bb in current_group]))
        rows.append((avg_y, current_group))
        current_group = [b]
rows.append((int(np.mean([bb[1] for bb in current_group])), current_group))

# === SBD/MDT HEADER ANALYSIS ===
print("=== SBD/MDT HEADER (right side, y < 0.30) ===")
header_right_rows = []
for y, row_bubbles in rows:
    if y < h * 0.30:
        right_bubbles = [(b[0], b[1]) for b in row_bubbles if b[0] > w * 0.65]
        if len(right_bubbles) >= 6:
            x_vals = sorted(set([int(round(b[0]/5)*5) for b in right_bubbles]))
            header_right_rows.append((y, x_vals))
            print(f"  y={y:4d} ({y/h:.3f}): {len(right_bubbles)} bubbles, x_groups={x_vals}")

# Identify SBD and MDT column groups from the first header row
if header_right_rows:
    all_x = header_right_rows[0][1]
    print(f"\n  All unique X positions (grouped by 5): {all_x}")
    
    # Find gaps to separate SBD from MDT
    gaps = []
    for i in range(1, len(all_x)):
        gap = all_x[i] - all_x[i-1]
        if gap > 60:
            gaps.append(i)
    
    if gaps:
        sbd_x = all_x[:gaps[0]]
        mdt_x = all_x[gaps[0]:]
        print(f"  SBD columns: {sbd_x} → x range {sbd_x[0]/w:.3f}-{sbd_x[-1]/w:.3f}")
        print(f"  MDT columns: {mdt_x} → x range {mdt_x[0]/w:.3f}-{mdt_x[-1]/w:.3f}")

    # Y range for header bubbles
    y_vals = [r[0] for r in header_right_rows]
    print(f"  Y range: {y_vals[0]}-{y_vals[-1]} ({y_vals[0]/h:.3f}-{y_vals[-1]/h:.3f})")
    print(f"  Row count: {len(y_vals)}")
    if len(y_vals) > 1:
        print(f"  Row spacing: {(y_vals[-1]-y_vals[0])/(len(y_vals)-1):.1f}px")

# === FC AREA ===
print("\n=== FC AREA ===")
fc_rows = [(y, bs) for y, bs in rows if 0.30*h <= y <= 0.53*h and len(bs) >= 15]
if fc_rows:
    y_vals = [r[0] for r in fc_rows]
    print(f"  Y range: {y_vals[0]}-{y_vals[-1]} ({y_vals[0]/h:.3f}-{y_vals[-1]/h:.3f})")
    print(f"  Row count: {len(y_vals)}")
    
    # Get columns from a representative row
    _, rep_bs = fc_rows[len(fc_rows)//2]
    x_vals = sorted([b[0] for b in rep_bs])
    
    # Cluster x values
    x_clusters = []
    current = [x_vals[0]]
    for x in x_vals[1:]:
        if x - current[-1] < 15:
            current.append(x)
        else:
            x_clusters.append(int(np.mean(current)))
            current = [x]
    x_clusters.append(int(np.mean(current)))
    
    # Group by large gaps (>100px)
    groups = []
    current_g = [x_clusters[0]]
    for x in x_clusters[1:]:
        if x - current_g[-1] > 100:
            groups.append(current_g)
            current_g = [x]
        else:
            current_g.append(x)
    groups.append(current_g)
    
    print(f"  Column groups ({len(groups)} groups):")
    for i, g in enumerate(groups):
        x1_pct = (g[0]-20)/w
        x2_pct = (g[-1]+20)/w
        print(f"    Group {i+1}: cols={g}, x_range={x1_pct:.3f}-{x2_pct:.3f}, {len(g)} cols")
    
    y1_pct = (y_vals[0]-20)/h
    y2_pct = (y_vals[-1]+20)/h
    print(f"  FC region: y1={y1_pct:.3f}, y2={y2_pct:.3f}")

# === TF AREA ===
print("\n=== TF AREA ===")
tf_rows = [(y, bs) for y, bs in rows if 0.54*h <= y <= 0.67*h and len(bs) >= 15]
if tf_rows:
    y_vals = [r[0] for r in tf_rows]
    print(f"  Y range: {y_vals[0]}-{y_vals[-1]} ({y_vals[0]/h:.3f}-{y_vals[-1]/h:.3f})")
    print(f"  Row count: {len(y_vals)}")
    
    _, rep_bs = tf_rows[len(tf_rows)//2]
    x_vals = sorted([b[0] for b in rep_bs])
    
    x_clusters = []
    current = [x_vals[0]]
    for x in x_vals[1:]:
        if x - current[-1] < 15:
            current.append(x)
        else:
            x_clusters.append(int(np.mean(current)))
            current = [x]
    x_clusters.append(int(np.mean(current)))
    
    groups = []
    current_g = [x_clusters[0]]
    for x in x_clusters[1:]:
        if x - current_g[-1] > 100:
            groups.append(current_g)
            current_g = [x]
        else:
            current_g.append(x)
    groups.append(current_g)
    
    print(f"  Column groups ({len(groups)} groups):")
    for i, g in enumerate(groups):
        x1_pct = (g[0]-20)/w
        x2_pct = (g[-1]+20)/w
        print(f"    Group {i+1}: cols={g}, x_range={x1_pct:.3f}-{x2_pct:.3f}, {len(g)} cols")
    
    y1_pct = (y_vals[0]-20)/h
    y2_pct = (y_vals[-1]+20)/h
    print(f"  TF region: y1={y1_pct:.3f}, y2={y2_pct:.3f}")

# === DG AREA ===
print("\n=== DG AREA ===")
dg_rows = [(y, bs) for y, bs in rows if y > 0.79*h and len(bs) >= 20]
if dg_rows:
    y_vals = [r[0] for r in dg_rows]
    print(f"  Y range: {y_vals[0]}-{y_vals[-1]} ({y_vals[0]/h:.3f}-{y_vals[-1]/h:.3f})")
    print(f"  Row count: {len(y_vals)} (need 12 for digits 0-9, comma, minus)")
    
    _, rep_bs = dg_rows[len(dg_rows)//2]
    x_vals = sorted([b[0] for b in rep_bs])
    
    x_clusters = []
    current = [x_vals[0]]
    for x in x_vals[1:]:
        if x - current[-1] < 15:
            current.append(x)
        else:
            x_clusters.append(int(np.mean(current)))
            current = [x]
    x_clusters.append(int(np.mean(current)))
    
    groups = []
    current_g = [x_clusters[0]]
    for x in x_clusters[1:]:
        if x - current_g[-1] > 90:
            groups.append(current_g)
            current_g = [x]
        else:
            current_g.append(x)
    groups.append(current_g)
    
    print(f"  Column groups ({len(groups)} groups):")
    for i, g in enumerate(groups):
        x1_pct = (g[0]-20)/w
        x2_pct = (g[-1]+20)/w
        print(f"    Group {i+1}: cols={g}, x_range={x1_pct:.3f}-{x2_pct:.3f}, {len(g)} cols")
    
    y1_pct = (y_vals[0]-20)/h
    y2_pct = (y_vals[-1]+20)/h
    print(f"  DG region: y1={y1_pct:.3f}, y2={y2_pct:.3f}")

# Also test with a second image to check consistency
print("\n\n=== CONSISTENCY CHECK: PhieuQG.0001 ===")
img2 = load_image('anh/test_imga/PhieuQG.0001.jpg')
corners2 = find_sheet_by_ink(img2)
if corners2 is not None:
    warped2 = warp_perspective(img2, corners2, 1800, 2500)
    warped2, _ = deskew(warped2)
    warped2 = cv2.resize(warped2, (1800, 2500))
    gray2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY)
    _, thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bubbles2 = []
    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if 200 < area < 4000:
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / bh if bh > 0 else 0
            if 0.4 < aspect < 2.5 and bw > 8 and bh > 8:
                cx, cy = x + bw // 2, y + bh // 2
                bubbles2.append((cx, cy))
    
    # Check right-side header columns
    right_header = [(cx, cy) for cx, cy in bubbles2 if cx > w*0.65 and cy < h*0.30]
    if right_header:
        x_vals = sorted(set([int(round(b[0]/5)*5) for b in right_header]))
        print(f"  Right header X groups: {x_vals}")
    
    # Check FC area rows
    fc_bubbles2 = [(cx, cy) for cx, cy in bubbles2 if 0.30*h <= cy <= 0.53*h]
    if fc_bubbles2:
        y_vals = sorted(set([int(round(b[1]/10)*10) for b in fc_bubbles2]))
        fc_y_vals = [y for y in y_vals if sum(1 for cx,cy in fc_bubbles2 if abs(cy-y)<15) >= 10]
        print(f"  FC rows (y): {fc_y_vals}")
