"""Full pipeline test with calibrated config."""
import cv2
import json
from core import process_image

config = json.load(open('config.json', 'r', encoding='utf-8'))

test_files = [
    'anh/test_imga/PhieuQG.0010.jpg',
    'anh/test_imga/PhieuQG.0001.jpg',
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
        
        fc_filled = {k: v for k, v in fc.items() if v}
        fc_empty = sum(1 for v in fc.values() if not v)
        print(f"  FC filled ({len(fc_filled)}/40): {fc_filled}")
        
        tf_filled = {k: v for k, v in tf.items() if v}
        print(f"  TF filled ({len(tf_filled)}/32): {tf_filled}")
        
        dg_filled = {k: v for k, v in dg.items() if v}
        print(f"  DG: {dg_filled}")
    else:
        print("  res: N/A")
    
    print(f"  errors: {output.get('err')}")
    
    if marked is not None:
        cv2.imwrite(f"result_{base}.jpg", marked)
    if debug_img is not None:
        cv2.imwrite(f"debug_{base}.jpg", debug_img)
    print(f"  Saved result + debug images")
