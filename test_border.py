"""Detect form border rectangle in warped images for consistent alignment."""
import cv2
import numpy as np
import glob
from core.preprocessor import load_image, preprocess
from core.detector import find_corners, warp_perspective, deskew

def find_form_border(warped):
    """Find the form's outer border rectangle in a warped image."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
    h, w = gray.shape[:2]
    
    # Binary - invert so content is white
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find all horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//4, 1))
    h_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    
    # Find all vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h//4))
    v_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    
    # Get horizontal line positions
    h_proj = np.sum(h_lines_img, axis=1)
    h_peaks = np.where(h_proj > w * 0.2 * 255)[0]  # at least 20% of width
    
    # Get vertical line positions
    v_proj = np.sum(v_lines_img, axis=0)
    v_peaks = np.where(v_proj > h * 0.2 * 255)[0]  # at least 20% of height
    
    # Cluster horizontal peaks
    h_clusters = []
    for y in h_peaks:
        if not h_clusters or y - h_clusters[-1][-1] > 5:
            h_clusters.append([y])
        else:
            h_clusters[-1].append(y)
    h_positions = sorted([int(np.mean(c)) for c in h_clusters])
    
    # Cluster vertical peaks  
    v_clusters = []
    for x in v_peaks:
        if not v_clusters or x - v_clusters[-1][-1] > 5:
            v_clusters.append([x])
        else:
            v_clusters[-1].append(x)
    v_positions = sorted([int(np.mean(c)) for c in v_clusters])
    
    return h_positions, v_positions


def analyze_image(image_path):
    name = image_path.split("\\")[-1].split("/")[-1]
    img = load_image(image_path)
    gray = preprocess(img, {})
    from core.preprocessor import preprocess_to_binary
    binary = preprocess_to_binary(img, {})
    corners, method = find_corners(img, binary, {})
    warped = warp_perspective(img, corners, 1800, 2500)
    warped, skew = deskew(warped)
    if abs(skew) > 0.3:
        warped = cv2.resize(warped, (1800, 2500), interpolation=cv2.INTER_CUBIC)
    
    h_pos, v_pos = find_form_border(warped)
    
    print(f"\n=== {name} ({method}) ===")
    print(f"  H-lines ({len(h_pos)}): {h_pos[:20]}")
    print(f"  H-lines %: {[round(y/2500, 3) for y in h_pos[:20]]}")
    print(f"  V-lines ({len(v_pos)}): {v_pos[:20]}")
    print(f"  V-lines %: {[round(x/1800, 3) for x in v_pos[:20]]}")
    
    # Try to find form border (first/last strong lines)
    if len(h_pos) >= 2:
        top = h_pos[0]
        bottom = h_pos[-1]
        print(f"  Form Y range: {top} - {bottom} ({round(top/2500,3)} - {round(bottom/2500,3)})")
    if len(v_pos) >= 2:
        left = v_pos[0]
        right = v_pos[-1]
        print(f"  Form X range: {left} - {right} ({round(left/1800,3)} - {round(right/1800,3)})")
    
    return h_pos, v_pos


targets = ["PhieuQG.0001.jpg", "PhieuQG.0004.jpg", "PhieuQG.0010.jpg", 
           "PhieuQG.0020.jpg", "PhieuQG.0050.jpg"]
bmp = ["2025-02-28_67c1618d6e062_image1_thptthongnhata.BMP"]
all_results = {}
for name in targets + bmp:
    path = f"anh/test_imga/{name}"
    try:
        h, v = analyze_image(path)
        all_results[name] = (h, v)
    except Exception as e:
        print(f"\n=== {name}: ERROR {e} ===")
