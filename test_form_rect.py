"""Analyze form structure to find the printed form rectangle in original images.
Goal: find a consistent form border detection that works across all image types."""
import cv2
import numpy as np
import glob
from core.preprocessor import load_image

def find_form_rectangle(img):
    """Find the printed form rectangular border in the original image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]
    
    # Method 1: Find largest rectangular contour
    # Use adaptive threshold to handle different lighting
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 5)
    
    # Close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    best_rect = None
    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if area < 0.1 * h * w:  # At least 10% of image
            break
        
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) == 4:
            # Check if it's roughly rectangular
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            rect_area = rect[1][0] * rect[1][1]
            if rect_area > 0:
                solidity = area / rect_area
                if solidity > 0.7:
                    best_rect = approx.reshape(4, 2)
                    return best_rect, "contour_4pt", area / (h * w)
        
        # Try minAreaRect for non-perfect rectangles
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(int)
        rect_area = rect[1][0] * rect[1][1]
        if rect_area > 0 and area / rect_area > 0.7:
            return box, "minAreaRect", area / (h * w)
    
    # Method 2: Line-based detection
    # Find long horizontal and vertical lines
    edges = cv2.Canny(gray, 50, 150)
    
    # Horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//3, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    h_proj = np.sum(h_lines, axis=1) / 255
    h_peaks = np.where(h_proj > w * 0.3)[0]
    
    # Vertical lines  
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h//3))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    v_proj = np.sum(v_lines, axis=0) / 255
    v_peaks = np.where(v_proj > h * 0.3)[0]
    
    if len(h_peaks) > 0 and len(v_peaks) > 0:
        top = h_peaks[0]
        bottom = h_peaks[-1]
        left = v_peaks[0]
        right = v_peaks[-1]
        if bottom - top > h * 0.3 and right - left > w * 0.3:
            rect = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
            return rect, "lines", ((right-left)*(bottom-top)) / (h*w)
    
    return None, "failed", 0


def test_image(image_path):
    name = image_path.split("\\")[-1].split("/")[-1]
    img = load_image(image_path)
    h, w = img.shape[:2]
    
    rect, method, coverage = find_form_rectangle(img)
    
    print(f"\n=== {name} ({w}x{h}) ===")
    print(f"  Method: {method}, coverage: {coverage:.3f}")
    if rect is not None:
        # Normalize corners to percentage
        pts = rect.astype(float)
        pts[:, 0] /= w
        pts[:, 1] /= h
        for i, (px, py) in enumerate(pts):
            print(f"  Corner {i}: ({px:.3f}, {py:.3f})")
        
        # Compute form dimensions as % of image
        xs = pts[:, 0]
        ys = pts[:, 1]
        print(f"  Form X range: {xs.min():.3f} - {xs.max():.3f} ({(xs.max()-xs.min())*100:.1f}%)")
        print(f"  Form Y range: {ys.min():.3f} - {ys.max():.3f} ({(ys.max()-ys.min())*100:.1f}%)")
    else:
        print("  No rectangle found!")


targets = ["PhieuQG.0001.jpg", "PhieuQG.0004.jpg", "PhieuQG.0010.jpg",
           "PhieuQG.0020.jpg", "PhieuQG.0050.jpg"]
bmp = ["2025-02-28_67c1618d6e062_image1_thptthongnhata.BMP"]
for name in targets + bmp:
    path = f"anh/test_imga/{name}"
    try:
        test_image(path)
    except Exception as e:
        print(f"\n=== {name}: ERROR {e} ===")
        import traceback; traceback.print_exc()
