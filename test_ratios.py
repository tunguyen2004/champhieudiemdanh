"""Diagnose fill ratios for SBD and FC to tune extraction."""
import cv2
from core import load_config
from core.preprocessor import load_image, preprocess, preprocess_to_binary
from core.detector import find_corners, warp_perspective, deskew, align_to_template
from core.extractor import get_bubble_rect, compute_fill_ratio

cfg = load_config()
img = load_image('anh/test_imga/PhieuQG.0010.jpg')
gray = preprocess(img, cfg)
binary = preprocess_to_binary(img, cfg)
corners, method = find_corners(img, binary, cfg)
warped = warp_perspective(img, corners, 1800, 2500)
warped, skew = deskew(warped)
if abs(skew) > 0.3:
    warped = cv2.resize(warped, (1800, 2500))
warped, aligned = align_to_template(warped, cfg)
warped_binary = preprocess_to_binary(warped, cfg)

h, w = warped_binary.shape[:2]
margin = cfg.get('bubble_margin', 0.15)

# SBD
sbd = cfg['regions']['sbd']
print("=== SBD Fill Ratios ===")
header = "       " + "  ".join(f"Col{c}" for c in range(6))
print(header)
for row in range(10):
    vals = []
    for col in range(6):
        rect = get_bubble_rect(sbd, row, col, w, h, margin)
        ratio = compute_fill_ratio(warped_binary, rect)
        vals.append(f"{ratio:.3f}")
    print(f"Row {row}: {'  '.join(vals)}")

# FC Group 1
print("\n=== FC Group 1 (Q1-10) ===")
fc_g = cfg['regions']['fc']['groups'][0]
for row in range(10):
    vals = []
    for col in range(4):
        rect = get_bubble_rect(fc_g, row, col, w, h, margin)
        ratio = compute_fill_ratio(warped_binary, rect)
        vals.append(f"{ratio:.3f}")
    q = row + 1
    print(f"Q{q:>2}: A={vals[0]}  B={vals[1]}  C={vals[2]}  D={vals[3]}")

# FC Group 4 (Q31-40)
print("\n=== FC Group 4 (Q31-40) ===")
fc_g4 = cfg['regions']['fc']['groups'][3]
for row in range(10):
    vals = []
    for col in range(4):
        rect = get_bubble_rect(fc_g4, row, col, w, h, margin)
        ratio = compute_fill_ratio(warped_binary, rect)
        vals.append(f"{ratio:.3f}")
    q = row + 31
    print(f"Q{q:>2}: A={vals[0]}  B={vals[1]}  C={vals[2]}  D={vals[3]}")

# Save warped binary for visual check
cv2.imwrite('debug/warped_binary_0010.jpg', warped_binary)
cv2.imwrite('debug/warped_0010.jpg', warped)
print("\nSaved debug/warped_binary_0010.jpg and debug/warped_0010.jpg")
