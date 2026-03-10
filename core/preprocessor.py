"""
Module tiền xử lý ảnh phiếu trắc nghiệm.
Pipeline: Grayscale → CLAHE → GaussianBlur → AdaptiveThreshold / Canny
"""
import cv2
import numpy as np


def load_image(path):
    """Đọc ảnh từ đường dẫn, hỗ trợ đường dẫn Unicode."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {path}")
    return img


def to_grayscale(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def enhance_contrast(gray, clip_limit=2.0, tile_size=(8, 8)):
    """CLAHE - cân bằng tương phản cục bộ, tốt hơn histogram equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)


def denoise(gray, kernel_size=(5, 5)):
    """Gaussian blur để khử nhiễu."""
    return cv2.GaussianBlur(gray, kernel_size, 0)


def to_binary(gray, block_size=11, c=2):
    """Adaptive threshold - ngưỡng hóa thích ứng, xử lý ánh sáng không đều."""
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c
    )


def get_edges(gray, low=50, high=150):
    """Canny Edge Detection - làm nổi bật đường viền."""
    return cv2.Canny(gray, low, high)


def preprocess(img, config=None):
    """Pipeline tiền xử lý đầy đủ: Gray → CLAHE → Blur → trả về ảnh gray đã xử lý."""
    cfg = config.get("preprocessing", {}) if config else {}

    gray = to_grayscale(img)

    clip = cfg.get("clahe_clip_limit", 2.0)
    tile = tuple(cfg.get("clahe_tile_size", [8, 8]))
    enhanced = enhance_contrast(gray, clip, tile)

    kernel = tuple(cfg.get("gaussian_kernel", [5, 5]))
    blurred = denoise(enhanced, kernel)

    return blurred


def preprocess_to_binary(img, config=None):
    """Pipeline đầy đủ đến binary: dùng cho bubble detection."""
    cfg = config.get("preprocessing", {}) if config else {}
    gray = preprocess(img, config)
    block = cfg.get("adaptive_block_size", 11)
    c = cfg.get("adaptive_c", 2)
    return to_binary(gray, block, c)


def preprocess_to_edges(img, config=None):
    """Pipeline đầy đủ đến edge: dùng cho contour detection."""
    cfg = config.get("preprocessing", {}) if config else {}
    gray = preprocess(img, config)
    low = cfg.get("canny_low", 50)
    high = cfg.get("canny_high", 150)
    return get_edges(gray, low, high)
