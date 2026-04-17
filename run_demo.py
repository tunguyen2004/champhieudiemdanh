"""Demo script - chạy pipeline trên 3 ảnh test."""
import json
import cv2
from core import load_config, process_image

config = load_config("config.json")

test_files = [
    "anh/test_imga/PhieuQG.0009.jpg",
    "anh/test_imga/PhieuQG.0010.jpg",
    "anh/test_imga/PhieuQG.0050.jpg",
]

for img_path in test_files:
    print(f"\n{'='*60}")
    print(f"FILE: {img_path}")
    print(f"{'='*60}")
    output, marked, debug_img = process_image(img_path, config, debug=True)

    method = output.get("_detection_method", "")
    sbd = output.get("sbd", "?")
    mdt = output.get("mdt", "?")
    print(f"Method: {method}")
    print(f"SBD: {sbd}")
    print(f"MDT: {mdt}")
    print(f"Skew: {output.get('_skew_angle', 0)}")

    # FC
    fc = output.get("res", {}).get("fc", {})
    fc_filled = sum(1 for v in fc.values() if v)
    print(f"FC: {fc_filled}/40 answered")

    # TF detail
    tf = output.get("res", {}).get("tf", {})
    tf_count = 0
    for q, qdata in tf.items():
        if isinstance(qdata, dict):
            for sub, ans in qdata.items():
                if ans:
                    tf_count += 1
    print(f"TF: {tf_count}/32 answered")
    for q_num in sorted(tf.keys(), key=lambda x: int(x)):
        qdata = tf[q_num]
        if isinstance(qdata, dict):
            parts = []
            for sub in ["a", "b", "c", "d"]:
                ans = qdata.get(sub, [])
                if ans:
                    labels = ["D", "S"]
                    txt = ",".join(labels[a] for a in ans if a < len(labels))
                else:
                    txt = "_"
                parts.append(f"{sub}={txt}")
            print(f"  Cau {q_num}: {' | '.join(parts)}")

    # DG
    dg = output.get("res", {}).get("dg", {})
    print(f"DG:")
    for q in sorted(dg.keys(), key=lambda x: int(x)):
        print(f"  Cau {q}: '{dg[q]}'")

    # Warnings
    warn = output.get("warn", "")
    if warn:
        print(f"WARNINGS: {warn}")
    errs = output.get("err", [])
    if errs:
        print(f"ERRORS: {errs}")

    # Save results
    if marked is not None:
        fname = img_path.split("/")[-1]
        out_path = f"anh/demo_result_{fname}"
        cv2.imwrite(out_path, marked)
        print(f"Marked: {out_path} shape={marked.shape}")
    if debug_img is not None:
        fname = img_path.split("/")[-1]
        dbg_path = f"anh/demo_debug_{fname}"
        cv2.imwrite(dbg_path, debug_img)
        print(f"Debug: {dbg_path}")

print("\nDone!")
