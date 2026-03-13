"""Quick test: verify auto-calibration works for SBD and TF."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_config, process_image

config = load_config("config.json")
test_imgs = [f"anh/test_imga/PhieuQG.{str(i).zfill(4)}.jpg" for i in range(1, 4)]

lines = []
for img in test_imgs:
    if not os.path.exists(img):
        continue
    output, marked, debug = process_image(img, config, debug=True)
    if output.get("err"):
        lines.append(f"{os.path.basename(img)}: ERR={output['err']}")
        continue
    
    sbd = output["sbd"]
    mdt = output["mdt"]
    tf = output["res"]["tf"]
    
    tf_groups = config["regions"]["tf"]["groups"]
    sub_labels = ["a","b","c","d"]
    tf_labels = config["regions"]["tf"]["labels"]
    
    tf_lines = []
    for g_idx, grp in enumerate(tf_groups):
        parts = []
        for r_idx, q_num in enumerate(grp.get("questions",[])):
            ans = tf.get(str(q_num), [])
            sub = sub_labels[r_idx]
            txt = ",".join(tf_labels[a] for a in ans if a < len(tf_labels)) if ans else "-"
            parts.append(f"{sub}={txt}")
        tf_lines.append(f"C{g_idx+1}: {' '.join(parts)}")
    
    lines.append(f"{os.path.basename(img)}: SBD={sbd} MDT={mdt}")
    lines.append(f"  TF: {' | '.join(tf_lines)}")
    
    if output.get("warn"):
        lines.append(f"  Warn: {output['warn']}")

with open("test_result.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Written to test_result.txt")
