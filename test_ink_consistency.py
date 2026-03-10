"""Test ink detection consistency across multiple images."""
import cv2
import json
import numpy as np
from core.preprocessor import load_image, preprocess_to_binary
from core.detector import find_sheet_by_ink, find_corners, warp_perspective, deskew

config = json.load(open('config.json', 'r', encoding='utf-8'))

test_files = [
    'anh/test_imga/PhieuQG.0001.jpg',
    'anh/test_imga/PhieuQG.0010.jpg',
    'anh/test_imga/PhieuQG.0020.jpg',
    'anh/test_imga/PhieuQG.0030.jpg',
    'anh/test_imga/PhieuQG.0050.jpg',
    'anh/test_imga/2025-02-28_67c1618d6e062_image1_thptthongnhata.BMP',
]

for f in test_files:
    img = load_image(f)
    
    # Force ink detection
    ink_corners = find_sheet_by_ink(img)
    
    # Also get normal detection
    binary = preprocess_to_binary(img, config)
    normal_corners, method = find_corners(img, binary, config)
    
    base = f.split('/')[-1].rsplit('.', 1)[0]
    print(f"\n=== {base} (img={img.shape[1]}x{img.shape[0]}) ===")
    print(f"  Normal method: {method}")
    
    if ink_corners is not None:
        warped = warp_perspective(img, ink_corners, 1800, 2500)
        warped = cv2.resize(warped, (1800, 2500))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Check SBD position by looking at right-side header area
        sbd_region = thresh[int(2500*0.086):int(2500*0.293), int(1800*0.718):int(1800*0.835)]
        sbd_density = np.mean(sbd_region) / 255 * 100
        
        # FC area
        fc_region = thresh[int(2500*0.354):int(2500*0.528), int(1800*0.078):int(1800*0.223)]
        fc_density = np.mean(fc_region) / 255 * 100
        
        # Check if content at expected SBD position
        print(f"  INK warp - SBD area density: {sbd_density:.1f}%")
        print(f"  INK warp - FC G1 area density: {fc_density:.1f}%")
        print(f"  INK corners: TL={ink_corners[0].tolist()}, BR={ink_corners[2].tolist()}")
        
        # Save warped image
        cv2.imwrite(f"ink_warp_{base}.jpg", warped)
    else:
        print("  INK: Failed to detect")
    
    if method == "markers":
        warped_m = warp_perspective(img, normal_corners, 1800, 2500)
        warped_m = cv2.resize(warped_m, (1800, 2500))
        gray_m = cv2.cvtColor(warped_m, cv2.COLOR_BGR2GRAY)
        _, thresh_m = cv2.threshold(gray_m, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        sbd_region_m = thresh_m[int(2500*0.086):int(2500*0.293), int(1800*0.718):int(1800*0.835)]
        sbd_density_m = np.mean(sbd_region_m) / 255 * 100
        
        fc_region_m = thresh_m[int(2500*0.354):int(2500*0.528), int(1800*0.078):int(1800*0.223)]
        fc_density_m = np.mean(fc_region_m) / 255 * 100
        
        print(f"  MARKER warp - SBD area density: {sbd_density_m:.1f}%")
        print(f"  MARKER warp - FC G1 area density: {fc_density_m:.1f}%")
