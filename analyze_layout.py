"""Analyze warped image to find actual form regions for config calibration."""
import cv2
import numpy as np

# Load a warped debug image
img = cv2.imread("debug_warp_PhieuQG.0010.jpg")
if img is None:
    print("Image not found, using pipeline...")
    import json
    from core.preprocessor import load_image, preprocess_to_binary
    from core.detector import find_corners, warp_perspective, deskew
    config = json.load(open('config.json', 'r', encoding='utf-8'))
    img = load_image('anh/test_imga/PhieuQG.0010.jpg')
    binary = preprocess_to_binary(img, config)
    corners, method = find_corners(img, binary, config)
    img = warp_perspective(img, corners, 1800, 2500)
    img, _ = deskew(img)
    img = cv2.resize(img, (1800, 2500))

h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

print(f"Image size: {w}x{h}")
print()

# Horizontal projection (sum of black pixels per row)
h_proj = np.sum(binary, axis=1) / 255
# Vertical projection (sum of black pixels per column)
v_proj = np.sum(binary, axis=0) / 255

# Find horizontal bands (rows with lots of black = borders/lines)
print("=== Horizontal structure (significant rows) ===")
threshold_h = w * 0.3  # rows with >30% black pixels
in_band = False
band_start = 0
bands = []
for i in range(h):
    if h_proj[i] > threshold_h:
        if not in_band:
            band_start = i
            in_band = True
    else:
        if in_band:
            bands.append((band_start, i-1))
            in_band = False
if in_band:
    bands.append((band_start, h-1))

for start, end in bands:
    pct_start = start / h
    pct_end = end / h
    print(f"  Row {start:4d}-{end:4d} ({pct_start:.3f}-{pct_end:.3f})")

print()
print("=== Vertical structure (significant columns) ===")
threshold_v = h * 0.05  # columns with >5% black pixels
in_band = False
for i in range(w):
    if v_proj[i] > threshold_v:
        if not in_band:
            band_start = i
            in_band = True
    else:
        if in_band:
            pct_start = band_start / w
            pct_end = (i-1) / w
            if i - band_start > 5:  # skip thin lines
                print(f"  Col {band_start:4d}-{i-1:4d} ({pct_start:.3f}-{pct_end:.3f})")
            in_band = False

# Analyze specific rows for content density to detect regions
print()
print("=== Row content density (sampled every 50 rows) ===")
for y in range(0, h, 50):
    density = h_proj[y] / w * 100
    bar = "#" * int(density)
    if density > 5:
        print(f"  Row {y:4d} ({y/h:.3f}): {density:5.1f}% {bar}")

# Look at the image in grid sections
print()
print("=== Grid analysis (10x14 grid) ===")
grid_w = w // 10
grid_h = h // 14
for gy in range(14):
    row_str = ""
    for gx in range(10):
        region = binary[gy*grid_h:(gy+1)*grid_h, gx*grid_w:(gx+1)*grid_w]
        density = np.mean(region) / 255 * 100
        if density > 20:
            row_str += "##"
        elif density > 10:
            row_str += "@@"
        elif density > 5:
            row_str += ".."
        else:
            row_str += "  "
    y_pct = gy / 14
    print(f"  Row {gy:2d} ({y_pct:.2f}): [{row_str}]")
