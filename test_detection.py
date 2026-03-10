"""Test script for the new ink-based sheet detection."""
import cv2
import json
import numpy as np
from core.preprocessor import load_image, preprocess_to_binary
from core.detector import find_corners, warp_perspective, deskew

config = json.load(open('config.json', 'r', encoding='utf-8'))

test_files = [
    'anh/test_imga/PhieuQG.0001.jpg',
    'anh/test_imga/PhieuQG.0010.jpg',
    'anh/test_imga/PhieuQG.0050.jpg',
    'anh/test_imga/2025-02-28_67c1618d6e062_image1_thptthongnhata.BMP',
]

for f in test_files:
    img = load_image(f)
    binary = preprocess_to_binary(img, config)
    corners, method = find_corners(img, binary, config)
    
    base = f.split('/')[-1].rsplit('.', 1)[0]
    print(f"{base}: method={method}")
    print(f"  corners={corners.tolist()}")
    print(f"  img_shape={img.shape}")
    
    # Warp + deskew
    warped = warp_perspective(img, corners, 1800, 2500)
    warped, angle = deskew(warped)
    if abs(angle) > 0.3:
        warped = cv2.resize(warped, (1800, 2500))
    print(f"  skew_angle={angle:.2f}")
    
    cv2.imwrite(f"debug_warp_{base}.jpg", warped)
    print(f"  saved debug_warp_{base}.jpg")
    print()
