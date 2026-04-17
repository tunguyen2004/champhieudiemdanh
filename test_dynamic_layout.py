"""Quick smoke test for dynamic layout inference."""
import json
from copy import deepcopy

from core import process_image


def main():
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    config = deepcopy(config)
    config.setdefault("dynamic_layout", {})
    config["dynamic_layout"]["enabled"] = True

    test_files = [
        "anh/test_imga/PhieuQG.0001.jpg",
        "anh/test_imga/PhieuQG.0010.jpg",
        "anh/test_imga/PhieuQG.0050.jpg",
    ]

    for path in test_files:
        output, _, _ = process_image(path, config, debug=False)
        print(f"\n=== {path} ===")
        print("  method:", output.get("_detection_method"))
        print("  dynamic:", output.get("_layout_dynamic"))
        print("  mode:", output.get("_layout_mode"))
        print("  candidates:", output.get("_layout_candidate_count"))
        print("  components:", output.get("_layout_component_count"))
        print("  anchors:", output.get("_layout_anchor_count"))
        print("  sections_raw:", output.get("_layout_resolved_sections_raw"))
        print("  sections:", output.get("_layout_resolved_sections"))
        print("  sbd:", output.get("sbd"))
        print("  mdt:", output.get("mdt"))
        warn_text = (output.get("warn", "") or "").encode("ascii", "ignore").decode("ascii")
        print("  warn:", warn_text)


if __name__ == "__main__":
    main()
