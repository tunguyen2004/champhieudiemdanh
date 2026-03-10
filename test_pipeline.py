"""Test pipeline with template alignment on multiple images."""
import cv2
from core import load_config, process_image

cfg = load_config()

targets = ["PhieuQG.0001.jpg", "PhieuQG.0004.jpg", "PhieuQG.0010.jpg",
           "PhieuQG.0020.jpg", "PhieuQG.0050.jpg",
           "2025-02-28_67c1618d6e062_image1_thptthongnhata.BMP"]

for name in targets:
    path = f"anh/test_imga/{name}"
    result, marked, debug_img = process_image(path, cfg, debug=True)
    short = name.replace(".", "_").replace(" ", "_")[:20]

    method = result.get("_detection_method", "?")
    sbd = result.get("sbd", "?")
    mdt = result.get("mdt", "?")

    print(f"\n=== {name} ===")
    print(f"  method: {method}")
    print(f"  SBD: {sbd}")
    print(f"  MDT: {mdt}")

    res = result.get("res", {})
    if res:
        fc = res.get("fc", {})
        filled_fc = {k: v for k, v in fc.items() if v}
        print(f"  FC filled ({len(filled_fc)}/40): {filled_fc}")

        tf = res.get("tf", {})
        filled_tf = {k: v for k, v in tf.items() if v}
        print(f"  TF filled ({len(filled_tf)}/32): {filled_tf}")

        dg = res.get("dg", {})
        print(f"  DG: {dg}")

    print(f"  errors: {result.get('err', [])}")

    if marked is not None:
        cv2.imwrite(f"anh/result_{short}.jpg", marked)
    if debug_img is not None:
        cv2.imwrite(f"anh/debug_{short}.jpg", debug_img)
    print(f"  Saved result + debug images")
