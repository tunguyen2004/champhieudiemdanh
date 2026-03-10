"""Template-based alignment using ORB feature matching.
Strategy: 
1. Warp the clean template to 1800x2500 as reference
2. For each test image, after initial ink warp, align to template
3. Use homography refinement for precise positioning
"""
import cv2
import numpy as np
from core.preprocessor import load_image, preprocess, preprocess_to_binary
from core.detector import find_corners, warp_perspective, deskew

# Load template
template_path = r"anh/sample_image/Copy of PhieuTracNghiepTHPT2025.png"
template = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), cv2.IMREAD_COLOR)
print(f"Template original: {template.shape[1]}x{template.shape[0]}")

# The template is a full page. We need to find the answer grid area in it.
# First, warp template using form detection to get consistent reference
tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
tpl_binary = cv2.adaptiveThreshold(tpl_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 5)
# Find the form rectangle in the template
tpl_contours, _ = cv2.findContours(tpl_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
tpl_contours = sorted(tpl_contours, key=cv2.contourArea, reverse=True)

for cnt in tpl_contours[:5]:
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    rect = cv2.minAreaRect(cnt)
    print(f"  Contour: area={area:.0f} ({area/(template.shape[0]*template.shape[1])*100:.1f}%), "
          f"approx_pts={len(approx)}, rect_size={rect[1]}")

# Use ink detection on template
_, tpl_thresh = cv2.threshold(tpl_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
tpl_closed = cv2.morphologyEx(tpl_thresh, cv2.MORPH_CLOSE, kernel)
tpl_points = cv2.findNonZero(tpl_closed)
tpl_rect = cv2.minAreaRect(tpl_points)
tpl_box = cv2.boxPoints(tpl_rect).astype(int)
print(f"\nTemplate ink rect: center={tpl_rect[0]}, size={tpl_rect[1]}, angle={tpl_rect[2]}")

# Order points
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect

tpl_corners = order_points(tpl_box.astype(float))
print(f"Template corners: {tpl_corners}")

# Warp template to reference size
W, H = 1800, 2500
dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype="float32")
M = cv2.getPerspectiveTransform(tpl_corners, dst)
tpl_warped = cv2.warpPerspective(template, M, (W, H))
cv2.imwrite("anh/template_warped.jpg", tpl_warped)
print(f"\nSaved warped template: {W}x{H}")

# Now try ORB feature matching
orb = cv2.ORB_create(nfeatures=5000)
tpl_warped_gray = cv2.cvtColor(tpl_warped, cv2.COLOR_BGR2GRAY)
kp_tpl, des_tpl = orb.detectAndCompute(tpl_warped_gray, None)
print(f"Template keypoints: {len(kp_tpl)}")

# Test on each image
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

targets = ["PhieuQG.0001.jpg", "PhieuQG.0004.jpg", "PhieuQG.0010.jpg",
           "PhieuQG.0020.jpg", "PhieuQG.0050.jpg",
           "2025-02-28_67c1618d6e062_image1_thptthongnhata.BMP"]

for name in targets:
    path = f"anh/test_imga/{name}"
    try:
        img = load_image(path)
        gray = preprocess(img, {})
        binary = preprocess_to_binary(img, {})
        corners, method = find_corners(img, binary, {})
        warped = warp_perspective(img, corners, W, H)
        warped, skew = deskew(warped)
        if abs(skew) > 0.3:
            warped = cv2.resize(warped, (W, H), interpolation=cv2.INTER_CUBIC)
        
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        kp_img, des_img = orb.detectAndCompute(warped_gray, None)
        
        if des_img is None or des_tpl is None:
            print(f"\n=== {name}: No descriptors ===")
            continue
        
        # Match
        matches = bf.knnMatch(des_img, des_tpl, k=2)
        
        # Ratio test
        good = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        
        print(f"\n=== {name} ({method}) ===")
        print(f"  Keypoints: {len(kp_img)}, Good matches: {len(good)}")
        
        if len(good) >= 10:
            src_pts = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_tpl[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            Mh, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = mask.ravel().sum()
            print(f"  Homography inliers: {inliers}/{len(good)}")
            
            if Mh is not None:
                # Apply homography to align warped image with template
                aligned = cv2.warpPerspective(warped, Mh, (W, H))
                cv2.imwrite(f"anh/aligned_{name.split('.')[0]}.jpg", aligned)
                
                # Show the transform effect
                corners_in = np.float32([[0,0],[W,0],[W,H],[0,H]]).reshape(-1,1,2)
                corners_out = cv2.perspectiveTransform(corners_in, Mh)
                print(f"  Mapped corners:")
                for i, pt in enumerate(corners_out.reshape(-1, 2)):
                    print(f"    {i}: ({pt[0]:.0f}, {pt[1]:.0f}) -> offset ({pt[0]-corners_in[i][0][0]:.0f}, {pt[1]-corners_in[i][0][1]:.0f})")
        else:
            print(f"  Not enough matches for homography")
    
    except Exception as e:
        print(f"\n=== {name}: ERROR ===")
        import traceback; traceback.print_exc()
