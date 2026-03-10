"""Find actual bubble positions in the aligned image to calibrate config."""
import cv2
import numpy as np
from core.preprocessor import load_image, preprocess, preprocess_to_binary
from core.detector import find_corners, warp_perspective, deskew, align_to_template
import json

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

W, H = 1800, 2500

# Process image
img = load_image("anh/test_imga/PhieuQG.0010.jpg")
gray = preprocess(img, config)
binary = preprocess_to_binary(img, config)
corners, method = find_corners(img, binary, config)
warped = warp_perspective(img, corners, W, H)
warped, skew = deskew(warped)
if abs(skew) > 0.3:
    warped = cv2.resize(warped, (W, H), interpolation=cv2.INTER_CUBIC)
aligned, ok = align_to_template(warped, config)
print(f"Aligned: {ok}")

aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

# Find circles in key regions
# SBD region (right side, upper area)
# Search in a broad area: x=1200-1800, y=100-800
for region_name, rx1, ry1, rx2, ry2 in [
    ("SBD_area", 1200, 100, 1800, 750),
    ("FC_area", 100, 800, 1750, 1400),
    ("TF_area", 100, 1350, 1750, 1750),
    ("DG_area", 100, 1750, 1750, 2450),
]:
    roi = aligned_gray[ry1:ry2, rx1:rx2]
    circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, 1, 10,
                                param1=50, param2=20,
                                minRadius=5, maxRadius=18)
    
    print(f"\n=== {region_name} ({rx1},{ry1})-({rx2},{ry2}) ===")
    if circles is not None:
        circles_abs = np.round(circles[0, :]).astype(int)
        # Convert to absolute coordinates
        circles_abs[:, 0] += rx1
        circles_abs[:, 1] += ry1
        
        # Cluster by Y
        ys = sorted(set(circles_abs[:, 1]))
        y_clusters = []
        for y in ys:
            if not y_clusters or y - y_clusters[-1][-1] > 12:
                y_clusters.append([y])
            else:
                y_clusters[-1].append(y)
        
        rows = [(int(np.mean(c)), len(c)) for c in y_clusters if len(c) >= 2]
        print(f"  {len(circles_abs)} circles, {len(rows)} rows (with 2+ circles)")
        
        for y_mean, count in rows[:15]:
            row_circles = circles_abs[np.abs(circles_abs[:, 1] - y_mean) < 15]
            xs = sorted(row_circles[:, 0])
            print(f"  y={y_mean} ({y_mean/H:.4f}): {count}c, x=[{xs[0]}-{xs[-1]}] ({xs[0]/W:.3f}-{xs[-1]/W:.3f})")
            if count <= 10:
                print(f"    xs: {list(xs)}")
                print(f"    as%: {[round(x/W, 3) for x in xs]}")
    else:
        print("  No circles found")
