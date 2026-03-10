"""Precisely locate bubble grid regions on the warped template image."""
import cv2
import numpy as np

img = cv2.imread("debug_template_warped.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
h, w = thresh.shape

# Find all small circular/square blobs (bubbles)
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

bubbles = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 100 < area < 3000:  # bubble-sized
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        if 0.5 < aspect < 2.0:  # roughly circular/square
            cx, cy = x + bw // 2, y + bh // 2
            bubbles.append((cx, cy, bw, bh, area))

print(f"Found {len(bubbles)} potential bubbles")

# Cluster bubbles by Y position to find rows
bubbles.sort(key=lambda b: b[1])  # sort by Y

# Find distinct Y rows (bubbles at similar Y = same row)
y_values = [b[1] for b in bubbles]
y_clusters = []
if y_values:
    current_cluster = [y_values[0]]
    for y in y_values[1:]:
        if y - current_cluster[-1] < 15:  # same row
            current_cluster.append(y)
        else:
            y_clusters.append(int(np.mean(current_cluster)))
            current_cluster = [y]
    y_clusters.append(int(np.mean(current_cluster)))

print(f"\nFound {len(y_clusters)} distinct bubble rows")
print("\n=== Bubble rows (Y positions) ===")
for i, y in enumerate(y_clusters):
    # Count bubbles in this row
    row_bubbles = [b for b in bubbles if abs(b[1] - y) < 15]
    x_positions = sorted([b[0] for b in row_bubbles])
    count = len(row_bubbles)
    x_min = min(x_positions) if x_positions else 0
    x_max = max(x_positions) if x_positions else 0
    print(f"  Row {i:2d}: y={y:4d} ({y/h:.3f})  count={count:3d}  x_range=[{x_min:4d}-{x_max:4d}] ({x_min/w:.3f}-{x_max/w:.3f})")

# Also cluster by X position to find columns
print("\n=== Bubble columns (X positions) ===")
x_values = sorted([b[0] for b in bubbles])
x_clusters = []
if x_values:
    current_cluster = [x_values[0]]
    for x in x_values[1:]:
        if x - current_cluster[-1] < 12:
            current_cluster.append(x)
        else:
            x_clusters.append((int(np.mean(current_cluster)), len(current_cluster)))
            current_cluster = [x]
    x_clusters.append((int(np.mean(current_cluster)), len(current_cluster)))

# Only show columns with significant bubble counts
for i, (x, count) in enumerate(x_clusters):
    if count >= 3:
        print(f"  Col {i:2d}: x={x:4d} ({x/w:.3f})  count={count:3d}")

# Draw all detected bubbles for visual verification
vis = img.copy()
for cx, cy, bw, bh, area in bubbles:
    cv2.circle(vis, (cx, cy), max(bw, bh) // 2, (0, 0, 255), 1)
cv2.imwrite("debug_template_bubbles.jpg", vis)
print("\nSaved debug_template_bubbles.jpg")
