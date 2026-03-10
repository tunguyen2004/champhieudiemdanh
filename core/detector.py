"""
Module phát hiện phiếu trắc nghiệm và biến đổi phối cảnh.
Workflow: Tìm corner markers / contour lớn nhất → Warp Perspective → Deskew → Ảnh phẳng chuẩn
"""
import cv2
import numpy as np


def order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left: x+y nhỏ nhất
    rect[2] = pts[np.argmax(s)]      # bottom-right: x+y lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right: y-x nhỏ nhất
    rect[3] = pts[np.argmax(diff)]   # bottom-left: y-x lớn nhất
    return rect


def _validate_quad(pts, img_shape):
    """
    Kiểm tra 4 điểm có tạo thành hình chữ nhật hợp lệ không.
    - Diện tích >= 10% diện tích ảnh
    - Tỉ lệ cạnh trong khoảng hợp lý (0.4 - 2.5)
    - Các góc gần 90 độ
    """
    area = cv2.contourArea(pts.astype(np.float32))
    img_area = img_shape[0] * img_shape[1]
    if area < img_area * 0.10:
        return False

    # Kiểm tra tỉ lệ cạnh
    d01 = np.linalg.norm(pts[0] - pts[1])
    d12 = np.linalg.norm(pts[1] - pts[2])
    if min(d01, d12) == 0:
        return False
    ratio = max(d01, d12) / min(d01, d12)
    if ratio > 3.0:
        return False

    return True


def find_corner_markers(binary, config=None):
    """
    Tìm 4 marker đen ở 4 góc phiếu (hình vuông đen đặc).
    Lọc chặt hơn: chỉ chọn marker nằm ở vùng rìa ảnh (20% mỗi bên).
    """
    det_cfg = config.get("detection", {}) if config else {}
    min_area = det_cfg.get("marker_min_area", 500)
    max_area = det_cfg.get("marker_max_area", 20000)
    aspect_range = det_cfg.get("marker_aspect_range", [0.7, 1.4])
    solidity_min = det_cfg.get("marker_solidity_min", 0.80)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = binary.shape[:2]

    # Chỉ xét marker nằm ở 25% rìa ảnh
    margin_x = w_img * 0.25
    margin_y = h_img * 0.25

    markers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 0
        if not (aspect_range[0] <= aspect <= aspect_range[1]):
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < solidity_min:
            continue

        cx, cy = x + w // 2, y + h // 2

        # Marker phải nằm ở vùng rìa (gần 1 trong 4 góc)
        near_left = cx < margin_x
        near_right = cx > w_img - margin_x
        near_top = cy < margin_y
        near_bottom = cy > h_img - margin_y
        if not ((near_left or near_right) and (near_top or near_bottom)):
            continue

        markers.append((cx, cy, area))

    if len(markers) < 4:
        return None

    # Chọn 4 marker gần 4 góc ảnh nhất
    corners_ref = [(0, 0), (w_img, 0), (w_img, h_img), (0, h_img)]
    selected = []
    remaining = list(markers)
    for ref_x, ref_y in corners_ref:
        if not remaining:
            return None
        closest = min(remaining, key=lambda m: (m[0] - ref_x)**2 + (m[1] - ref_y)**2)
        selected.append([float(closest[0]), float(closest[1])])
        remaining.remove(closest)

    pts = order_points(np.array(selected, dtype=np.float32))

    if not _validate_quad(pts, binary.shape):
        return None

    return pts


def find_sheet_contour(gray):
    """
    Phương pháp dự phòng: tìm contour hình chữ nhật lớn nhất (= viền phiếu).
    Thử nhiều mức epsilon cho approxPolyDP.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:15]:
        # Thử nhiều mức epsilon
        for eps in [0.015, 0.02, 0.03, 0.04, 0.05]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                pts = order_points(approx.reshape(4, 2))
                if _validate_quad(pts, gray.shape):
                    return pts

    # Fallback: dùng minAreaRect trên contour lớn nhất
    if contours:
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        return order_points(box.astype(np.float32))

    h, w = gray.shape[:2]
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]])


def find_sheet_by_ink(img):
    """
    Tìm vùng phiếu dựa trên toàn bộ pixel mực (nội dung in).
    Dùng Otsu threshold → morphological close → minAreaRect trên tất cả non-zero pixels.
    Phương pháp robust hơn marker detection vì không phụ thuộc vào marker cụ thể.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Otsu threshold (đảo: mực đen = trắng)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological close để nối các vùng in gần nhau
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Tìm tất cả pixel non-zero
    points = cv2.findNonZero(closed)
    if points is None or len(points) < 1000:
        return None

    # minAreaRect cho bounding box xoay tối ưu
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    pts = order_points(box.astype(np.float32))

    # Validate
    h_img, w_img = gray.shape[:2]
    area = cv2.contourArea(pts)
    if area < h_img * w_img * 0.15:
        return None

    return pts


def find_corners(img, binary, config=None):
    """
    Tìm 4 góc phiếu - thử nhiều phương pháp theo thứ tự ưu tiên:
    1. Corner markers (nếu phiếu có 4 hình vuông đen ở 4 góc)
    2. Ink-based detection (dùng toàn bộ nội dung in để tìm vùng phiếu)
    3. Contour fallback (tìm contour hình chữ nhật lớn nhất)
    """
    # Phương pháp 1: Corner markers
    corners = find_corner_markers(binary, config)
    if corners is not None:
        return corners, "markers"

    # Phương pháp 2: Ink-based (robust nhất cho phiếu quét)
    corners = find_sheet_by_ink(img)
    if corners is not None:
        return corners, "ink"

    # Phương pháp 3: Contour fallback
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    corners = find_sheet_contour(gray)
    return corners, "contour"


def warp_perspective(img, corners, target_w, target_h):
    """Biến đổi phối cảnh - trải phẳng phiếu về kích thước chuẩn."""
    dst = np.float32([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ])
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(img, matrix, (target_w, target_h))
    return warped


def deskew(img):
    """
    Chỉnh nghiêng ảnh dựa trên HoughLines.
    Hoạt động với cả ảnh màu (BGR) và grayscale.
    Trả về (ảnh_đã_xoay, góc_xoay_độ).
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Phát hiện cạnh
    edges = cv2.Canny(gray, 50, 150)

    # Tìm đường thẳng - ưu tiên đường dài (đường viền phiếu)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                             minLineLength=min(gray.shape) // 5,
                             maxLineGap=15)

    if lines is None:
        return img, 0.0

    # Thu thập góc của đường gần ngang và gần dọc
    h_angles = []
    v_angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Đường gần ngang (±20°)
        if abs(angle) < 20:
            h_angles.append((angle, length))
        # Đường gần dọc (±70-110°)
        elif 70 < abs(angle) < 110:
            v_angle = angle - 90 if angle > 0 else angle + 90
            v_angles.append((v_angle, length))

    # Tính góc trung bình có trọng số (ưu tiên đường dài)
    all_weighted = h_angles + v_angles
    if not all_weighted:
        return img, 0.0

    total_weight = sum(w for _, w in all_weighted)
    if total_weight == 0:
        return img, 0.0
    weighted_angle = sum(a * w for a, w in all_weighted) / total_weight

    # Chỉ xoay nếu góc nghiêng đáng kể (>0.3°)
    if abs(weighted_angle) < 0.3:
        return img, 0.0

    # Giới hạn góc xoay tối đa ±15° (phiếu quá nghiêng = lỗi khác)
    if abs(weighted_angle) > 15:
        return img, 0.0

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, weighted_angle, 1.0)

    # Tính kích thước ảnh mới để không bị cắt viền
    cos_a = abs(np.cos(np.radians(weighted_angle)))
    sin_a = abs(np.sin(np.radians(weighted_angle)))
    new_w = int(w * cos_a + h * sin_a)
    new_h = int(w * sin_a + h * cos_a)
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(img, rotation_matrix, (new_w, new_h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    # Crop lại về kích thước gốc (cắt viền trắng)
    if new_w > w or new_h > h:
        dx = (new_w - w) // 2
        dy = (new_h - h) // 2
        rotated = rotated[dy:dy+h, dx:dx+w]

    return rotated, weighted_angle


class TemplateAligner:
    """
    Căn chỉnh ảnh đã warp về tọa độ chuẩn bằng ORB feature matching với template.
    Singleton - chỉ load template 1 lần.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _init_template(self, template_path, target_w, target_h):
        """Load và warp template 1 lần."""
        if self._initialized:
            return

        template = cv2.imdecode(
            np.fromfile(template_path, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if template is None:
            self._initialized = True
            self._available = False
            return

        # Tìm vùng mực template bằng Otsu + minAreaRect
        tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        _, tpl_thresh = cv2.threshold(
            tpl_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        tpl_closed = cv2.morphologyEx(tpl_thresh, cv2.MORPH_CLOSE, kernel)
        tpl_points = cv2.findNonZero(tpl_closed)
        if tpl_points is None:
            self._initialized = True
            self._available = False
            return

        tpl_rect = cv2.minAreaRect(tpl_points)
        tpl_box = cv2.boxPoints(tpl_rect).astype(np.float32)
        tpl_corners = order_points(tpl_box)

        # Warp template về kích thước chuẩn
        dst = np.float32([
            [0, 0], [target_w - 1, 0],
            [target_w - 1, target_h - 1], [0, target_h - 1]
        ])
        M = cv2.getPerspectiveTransform(tpl_corners, dst)
        self._tpl_warped = cv2.warpPerspective(template, M, (target_w, target_h))

        # ORB features trên template
        self._orb = cv2.ORB_create(nfeatures=5000)
        tpl_warped_gray = cv2.cvtColor(self._tpl_warped, cv2.COLOR_BGR2GRAY)
        self._kp_tpl, self._des_tpl = self._orb.detectAndCompute(tpl_warped_gray, None)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._target_w = target_w
        self._target_h = target_h

        self._available = self._des_tpl is not None and len(self._des_tpl) > 50
        self._initialized = True

    def align(self, warped, template_path, target_w, target_h):
        """
        Căn chỉnh ảnh warped theo template.
        Returns: (aligned_image, success_bool)
        """
        self._init_template(template_path, target_w, target_h)

        if not self._available:
            return warped, False

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
        kp_img, des_img = self._orb.detectAndCompute(gray, None)

        if des_img is None or len(des_img) < 20:
            return warped, False

        # KNN matching + ratio test
        matches = self._bf.knnMatch(des_img, self._des_tpl, k=2)
        good = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 20:
            return warped, False

        src_pts = np.float32(
            [kp_img[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [self._kp_tpl[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        Mh, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if Mh is None:
            return warped, False

        inliers = mask.ravel().sum()
        if inliers < 15:
            return warped, False

        aligned = cv2.warpPerspective(warped, Mh, (target_w, target_h))
        return aligned, True


def align_to_template(warped, config):
    """
    Căn chỉnh ảnh warped theo template sử dụng ORB feature matching.
    Returns: (aligned_image, success_bool)
    """
    template_path = config.get(
        "template_path",
        "anh/sample_image/Copy of PhieuTracNghiepTHPT2025.png"
    )
    target_w = config.get("warp_width", 1800)
    target_h = config.get("warp_height", 2500)

    aligner = TemplateAligner()
    return aligner.align(warped, template_path, target_w, target_h)
