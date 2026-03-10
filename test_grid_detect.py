"""Find the answer grid rectangle INSIDE the warped image for re-alignment.
Strategy: ink warp gets the page into frame, then we find the grid border for precise alignment."""
import cv2
import numpy as np
from core.preprocessor import load_image, preprocess, preprocess_to_binary
from core.detector import find_corners, warp_perspective, deskew

def find_grid_in_warped(warped):
    """Find the answer grid rectangle in a warped image.
    The grid has thick black borders forming a large rectangle."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
    h, w = gray.shape[:2]
    
    # Use morphological line detection
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 5)
    
    # Detect long horizontal lines (at least 40% of width)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w*0.4), 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    
    # Detect long vertical lines (at least 40% of height)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h*0.4)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    
    # Project to find line positions
    h_proj = np.sum(h_lines, axis=1) / 255
    v_proj = np.sum(v_lines, axis=0) / 255
    
    # Find significant horizontal lines (at least 30% of width coverage)
    h_thresh = w * 0.3
    h_peaks = np.where(h_proj > h_thresh)[0]
    
    # Find significant vertical lines
    v_thresh = h * 0.3
    v_peaks = np.where(v_proj > v_thresh)[0]
    
    # Cluster horizontal lines
    h_clusters = []
    for y in h_peaks:
        if not h_clusters or y - h_clusters[-1][-1] > 10:
            h_clusters.append([y])
        else:
            h_clusters[-1].append(y)
    h_positions = [int(np.mean(c)) for c in h_clusters]
    
    # Cluster vertical lines
    v_clusters = []
    for x in v_peaks:
        if not v_clusters or x - v_clusters[-1][-1] > 10:
            v_clusters.append([x])
        else:
            v_clusters[-1].append(x)
    v_positions = [int(np.mean(c)) for c in v_clusters]
    
    return h_positions, v_positions


def analyze_warped(image_path):
    name = image_path.split("\\")[-1].split("/")[-1]
    img = load_image(image_path)
    gray = preprocess(img, {})
    binary = preprocess_to_binary(img, {})
    corners, method = find_corners(img, binary, {})
    warped = warp_perspective(img, corners, 1800, 2500)
    warped, skew = deskew(warped)
    if abs(skew) > 0.3:
        warped = cv2.resize(warped, (1800, 2500), interpolation=cv2.INTER_CUBIC)
    
    h_pos, v_pos = find_grid_in_warped(warped)
    
    print(f"\n=== {name} ({method}) ===")
    print(f"  H-lines ({len(h_pos)}): {h_pos}")
    print(f"  H as %: {[round(y/2500, 3) for y in h_pos]}")
    print(f"  V-lines ({len(v_pos)}): {v_pos}")
    print(f"  V as %: {[round(x/1800, 3) for x in v_pos]}")
    
    # Identify potential grid boundaries
    if len(h_pos) >= 2 and len(v_pos) >= 2:
        # Grid top = first H-line, Grid bottom = last H-line
        # Grid left = first V-line, Grid right = last V-line
        grid_top = h_pos[0]
        grid_bottom = h_pos[-1]
        grid_left = v_pos[0]
        grid_right = v_pos[-1]
        grid_w = grid_right - grid_left
        grid_h = grid_bottom - grid_top
        print(f"  Grid box: ({grid_left},{grid_top}) to ({grid_right},{grid_bottom})")
        print(f"  Grid size: {grid_w}x{grid_h}")
        print(f"  Grid as %: x=[{grid_left/1800:.3f}-{grid_right/1800:.3f}], y=[{grid_top/2500:.3f}-{grid_bottom/2500:.3f}]")
        
        # Save the warped image with grid overlay for inspection
        vis = warped.copy()
        cv2.rectangle(vis, (grid_left, grid_top), (grid_right, grid_bottom), (0, 255, 0), 3)
        for y in h_pos:
            cv2.line(vis, (0, y), (1800, y), (0, 0, 255), 1)
        for x in v_pos:
            cv2.line(vis, (x, 0), (x, 2500), (255, 0, 0), 1)
        cv2.imwrite(f"anh/grid_detect_{name.split('.')[0]}.jpg", vis)
    
    return warped, h_pos, v_pos


targets = ["PhieuQG.0001.jpg", "PhieuQG.0004.jpg", "PhieuQG.0010.jpg",
           "PhieuQG.0020.jpg", "PhieuQG.0050.jpg"]
bmp = ["2025-02-28_67c1618d6e062_image1_thptthongnhata.BMP"]
for name in targets + bmp:
    path = f"anh/test_imga/{name}"
    try:
        analyze_warped(path)
    except Exception as e:
        print(f"\n=== {name}: ERROR ===")
        import traceback; traceback.print_exc()
