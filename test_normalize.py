"""Test normalize_content effect on different images."""
import cv2
import numpy as np
import glob
from core.preprocessor import load_image, preprocess_to_binary, preprocess
from core.detector import find_corners, warp_perspective, deskew

def test_normalize(image_path):
    img = load_image(image_path)
    gray = preprocess(img, {})
    binary = preprocess_to_binary(img, {})
    corners, method = find_corners(img, binary, {})
    warped = warp_perspective(img, corners, 1800, 2500)
    warped, skew = deskew(warped)
    if abs(skew) > 0.3:
        warped = cv2.resize(warped, (1800, 2500), interpolation=cv2.INTER_CUBIC)
    
    # Simulate normalize_content with debug
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h, w = gray_w.shape[:2]
    _, thresh = cv2.threshold(gray_w, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    points = cv2.findNonZero(closed)
    if points is None:
        print(f"  No content found!")
        return
    x, y, bw, bh = cv2.boundingRect(points)
    ratio_w = bw / w
    ratio_h = bh / h
    print(f"  method={method}, warped={w}x{h}")
    print(f"  content BB: x={x}, y={y}, w={bw}, h={bh}")
    print(f"  ratio: w={ratio_w:.3f}, h={ratio_h:.3f}")
    print(f"  normalize active: {not (ratio_w > 0.95 and ratio_h > 0.95)}")
    
    # What if we try to find the form border lines?
    # Find strong horizontal lines in top half
    edges = cv2.Canny(gray_w, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, minLineLength=500, maxLineGap=10)
    if lines is not None:
        h_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if angle < 5 or angle > 175:  # horizontal
                h_lines.append(y1)
        if h_lines:
            h_lines.sort()
            # Cluster lines
            clusters = []
            for y_val in h_lines:
                if not clusters or y_val - clusters[-1][-1] > 20:
                    clusters.append([y_val])
                else:
                    clusters[-1].append(y_val)
            major = [int(np.mean(c)) for c in clusters if len(c) >= 2]
            print(f"  Major H-lines (y): {major[:15]}")
            print(f"  As % of height: {[round(y_val/h, 3) for y_val in major[:15]]}")

test_images = sorted(glob.glob("anh/test_imga/PhieuQG.*.jpg"))
bmp_images = sorted(glob.glob("anh/test_imga/*.BMP"))
# Include 0001, 0004, 0010, 0050, BMP
targets = ["PhieuQG.0001.jpg", "PhieuQG.0004.jpg", "PhieuQG.0010.jpg", "PhieuQG.0050.jpg"]
all_images = [f"anh/test_imga/{t}" for t in targets] + bmp_images[:1]
for img_path in all_images:
    name = img_path.split("\\")[-1].split("/")[-1]
    print(f"\n=== {name} ===")
    test_normalize(img_path)
