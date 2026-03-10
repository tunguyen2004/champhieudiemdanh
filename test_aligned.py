"""Test extraction on template-aligned images."""
import cv2
import numpy as np
import json
from core.preprocessor import load_image, preprocess, preprocess_to_binary
from core.detector import find_corners, warp_perspective, deskew
from core.extractor import extract_all
from core.visualizer import visualize_results

# Load config
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

W, H = 1800, 2500

# Prepare template
template_path = r"anh/sample_image/Copy of PhieuTracNghiepTHPT2025.png"
template = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), cv2.IMREAD_COLOR)
tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
_, tpl_thresh = cv2.threshold(tpl_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
tpl_closed = cv2.morphologyEx(tpl_thresh, cv2.MORPH_CLOSE, kernel)
tpl_points = cv2.findNonZero(tpl_closed)
tpl_rect = cv2.minAreaRect(tpl_points)
tpl_box = cv2.boxPoints(tpl_rect).astype(float)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

tpl_corners = order_points(tpl_box)
dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype="float32")
M_tpl = cv2.getPerspectiveTransform(tpl_corners, dst)
tpl_warped = cv2.warpPerspective(template, M_tpl, (W, H))

# ORB on template
orb = cv2.ORB_create(nfeatures=5000)
tpl_warped_gray = cv2.cvtColor(tpl_warped, cv2.COLOR_BGR2GRAY)
kp_tpl, des_tpl = orb.detectAndCompute(tpl_warped_gray, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def align_to_template(warped):
    """Align a warped image to the template using ORB feature matching."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
    kp_img, des_img = orb.detectAndCompute(gray, None)
    
    if des_img is None or len(des_img) < 10:
        return warped, False
    
    matches = bf.knnMatch(des_img, des_tpl, k=2)
    good = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
    
    if len(good) < 20:
        return warped, False
    
    src_pts = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_tpl[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    Mh, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if Mh is None:
        return warped, False
    
    inliers = mask.ravel().sum()
    if inliers < 10:
        return warped, False
    
    aligned = cv2.warpPerspective(warped, Mh, (W, H))
    return aligned, True


targets = ["PhieuQG.0001.jpg", "PhieuQG.0004.jpg", "PhieuQG.0010.jpg",
           "PhieuQG.0020.jpg", "PhieuQG.0050.jpg",
           "2025-02-28_67c1618d6e062_image1_thptthongnhata.BMP"]

for name in targets:
    path = f"anh/test_imga/{name}"
    try:
        img = load_image(path)
        gray = preprocess(img, config)
        binary = preprocess_to_binary(img, config)
        corners, method = find_corners(img, binary, config)
        warped = warp_perspective(img, corners, W, H)
        warped, skew = deskew(warped)
        if abs(skew) > 0.3:
            warped = cv2.resize(warped, (W, H), interpolation=cv2.INTER_CUBIC)
        
        # Align to template
        aligned, success = align_to_template(warped)
        
        # Extract from aligned
        aligned_binary = preprocess_to_binary(aligned, config)
        results, errors, warnings = extract_all(aligned_binary, config)
        
        # Visualize
        marked = visualize_results(aligned, results, config)
        short = name.split('.')[0] if '.' in name else name[:20]
        cv2.imwrite(f"anh/aligned_result_{short}.jpg", marked)
        
        print(f"\n=== {name} ({method}, aligned={success}) ===")
        print(f"  SBD: {results.get('sbd', '?')}")
        print(f"  MDT: {results.get('mdt', '?')}")
        
        fc = results.get('fc', {})
        filled_fc = {k: v for k, v in fc.items() if v}
        print(f"  FC filled ({len(filled_fc)}/40): {filled_fc}")
        
        tf = results.get('tf', {})
        filled_tf = {k: v for k, v in tf.items() if v}
        print(f"  TF filled ({len(filled_tf)}/32): {filled_tf}")
        
        dg = results.get('dg', {})
        print(f"  DG: {dg}")
        print(f"  Errors: {errors}")
        
    except Exception as e:
        print(f"\n=== {name}: ERROR ===")
        import traceback; traceback.print_exc()
