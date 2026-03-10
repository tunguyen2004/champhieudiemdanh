"""Analyze the warped template to find exact bubble positions for config calibration."""
import cv2
import numpy as np

# Load the warped template
tpl = cv2.imread("anh/template_warped.jpg")
if tpl is None:
    print("ERROR: template_warped.jpg not found. Run test_template_match.py first.")
    exit(1)

print(f"Warped template: {tpl.shape}")
gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
h, w = gray.shape[:2]

# Binary
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find horizontal lines at different minimum lengths
for min_len_pct in [0.15, 0.30, 0.50]:
    min_len = int(w * min_len_pct)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    h_proj = np.sum(h_lines, axis=1) / 255
    h_peaks = np.where(h_proj > min_len * 0.5)[0]
    
    clusters = []
    for y_val in h_peaks:
        if not clusters or y_val - clusters[-1][-1] > 8:
            clusters.append([y_val])
        else:
            clusters[-1].append(y_val)
    positions = [int(np.mean(c)) for c in clusters]
    print(f"\nH-lines (min_len={min_len_pct*100:.0f}%={min_len}px): {len(positions)}")
    for y_val in positions:
        print(f"  y={y_val} ({y_val/h:.4f})")

print("\n---")
# Find vertical lines
for min_len_pct in [0.15, 0.30]:
    min_len = int(h * min_len_pct)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    v_proj = np.sum(v_lines, axis=0) / 255
    v_peaks = np.where(v_proj > min_len * 0.5)[0]
    
    clusters = []
    for x_val in v_peaks:
        if not clusters or x_val - clusters[-1][-1] > 8:
            clusters.append([x_val])
        else:
            clusters[-1].append(x_val)
    positions = [int(np.mean(c)) for c in clusters]
    print(f"\nV-lines (min_len={min_len_pct*100:.0f}%={min_len}px): {len(positions)}")
    for x_val in positions:
        print(f"  x={x_val} ({x_val/w:.4f})")

# Detect circles (bubbles)
print("\n=== Bubble detection ===")
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10,
                            param1=50, param2=25,
                            minRadius=5, maxRadius=18)
if circles is not None:
    circles = np.round(circles[0, :]).astype(int)
    print(f"Found {len(circles)} circles total")
    
    # Cluster by Y to find bubble rows
    ys = sorted([c[1] for c in circles])
    y_clusters = []
    for y_val in ys:
        if not y_clusters or y_val - y_clusters[-1][-1] > 12:
            y_clusters.append([y_val])
        else:
            y_clusters[-1].append(y_val)
    
    bubble_rows = [(int(np.mean(c)), len(c)) for c in y_clusters if len(c) >= 3]
    print(f"\nBubble rows ({len(bubble_rows)} rows with 3+ circles):")
    for y_mean, count in bubble_rows:
        # Find x positions of circles in this row
        row_circles = [c for c in circles if abs(c[1] - y_mean) < 15]
        xs = sorted([c[0] for c in row_circles])
        print(f"  y={y_mean} ({y_mean/h:.4f}): {count} bubbles, x=[{xs[0]}-{xs[-1]}] ({xs[0]/w:.3f}-{xs[-1]/w:.3f})")
        if count <= 20:
            print(f"    x positions: {xs}")
            print(f"    x as %: {[round(x/w, 3) for x in xs]}")
else:
    print("No circles found")
