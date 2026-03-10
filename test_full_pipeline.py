"""Full pipeline test with debug output."""
import cv2
import json
from core import process_image

config = json.load(open('config.json', 'r', encoding='utf-8'))

test_files = [
    'anh/test_imga/PhieuQG.0001.jpg',
    'anh/test_imga/PhieuQG.0010.jpg',
    'anh/test_imga/PhieuQG.0050.jpg',
]

for f in test_files:
    output, marked, debug_img = process_image(f, config, debug=True)
    base = f.split('/')[-1].rsplit('.', 1)[0]
    
    print(f"\n=== {base} ===")
    print(f"  method: {output.get('_detection_method')}")
    print(f"  skew_angle: {output.get('_skew_angle')}")
    print(f"  SBD: {output.get('sbd')}")
    print(f"  MDT: {output.get('mdt')}")
    if output.get('res'):
        fc = output['res']['fc']
        tf = output['res']['tf']
        dg = output['res']['dg']
        fc_items = list(fc.items())[:5] if isinstance(fc, dict) else fc[:5]
        tf_items = list(tf.items())[:4] if isinstance(tf, dict) else tf[:4]
        print(f"  FC (first 5): {fc_items}")
        print(f"  TF (first 4): {tf_items}")
        print(f"  DG: {dg}")
    else:
        print("  res: N/A")
    print(f"  errors: {output.get('err')}")
    
    if marked is not None:
        cv2.imwrite(f"result_{base}.jpg", marked)
    if debug_img is not None:
        cv2.imwrite(f"debug_{base}.jpg", debug_img)
    print(f"  Saved result + debug images")
