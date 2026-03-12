"""
Flask Web Application - Hệ thống chấm phiếu trắc nghiệm THPT 2025
Giao diện web cho phép upload ảnh, xử lý và hiển thị kết quả.
"""
import os
import json
import csv
import io
import uuid
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for

import cv2
from core import load_config, process_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
DEBUG_DIR = os.path.join(BASE_DIR, "debug")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """Xử lý 1 hoặc nhiều ảnh phiếu trắc nghiệm."""
    if "files" not in request.files:
        return render_template("index.html", error="Chưa chọn file ảnh")

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return render_template("index.html", error="Chưa chọn file ảnh")

    debug_mode = request.form.get("debug", "0") == "1"

    config = load_config(CONFIG_PATH)
    all_results = []

    for file in files:
        if not file or file.filename == "":
            continue
        if not allowed_file(file.filename):
            all_results.append({
                "org": file.filename,
                "err": [f"Định dạng không hỗ trợ: {file.filename}"],
                "res": None
            })
            continue

        # Lưu file upload
        safe_name = f"{uuid.uuid4().hex}_{file.filename}"
        upload_path = os.path.join(UPLOAD_DIR, safe_name)
        file.save(upload_path)

        # Xử lý
        output, marked, debug_img = process_image(upload_path, config, debug=debug_mode)

        # Lưu ảnh kết quả
        if marked is not None:
            marked_name = f"marked_{safe_name}.jpg"
            marked_path = os.path.join(RESULT_DIR, marked_name)
            cv2.imwrite(marked_path, marked)
            output["out"] = marked_name
            output["_marked_url"] = f"/result_image/{marked_name}"

        if debug_img is not None:
            debug_name = f"debug_{safe_name}.jpg"
            debug_path = os.path.join(DEBUG_DIR, debug_name)
            cv2.imwrite(debug_path, debug_img)
            output["_debug_url"] = f"/debug_image/{debug_name}"

        output["_original_name"] = file.filename
        all_results.append(output)

    # Lưu JSON kết quả
    result_id = uuid.uuid4().hex[:8]
    json_name = f"results_{result_id}.json"
    json_path = os.path.join(RESULT_DIR, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    return render_template("result.html",
                           results=all_results,
                           result_id=result_id,
                           json_name=json_name,
                           tf_groups=config["regions"]["tf"]["groups"],
                           tf_labels=config["regions"]["tf"]["labels"])


@app.route("/api/process", methods=["POST"])
def api_process():
    """API endpoint - trả về JSON."""
    if "file" not in request.files:
        return jsonify({"error": "Chưa có file"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "File không hợp lệ"}), 400

    config = load_config(CONFIG_PATH)

    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    upload_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(upload_path)

    output, marked, _ = process_image(upload_path, config)

    if marked is not None:
        marked_name = f"marked_{safe_name}.jpg"
        marked_path = os.path.join(RESULT_DIR, marked_name)
        cv2.imwrite(marked_path, marked)
        output["out"] = f"/result_image/{marked_name}"

    # Xóa trường nội bộ
    output.pop("_detection_method", None)

    return jsonify(output)


@app.route("/result_image/<filename>")
def result_image(filename):
    """Serve ảnh kết quả."""
    path = os.path.join(RESULT_DIR, filename)
    if not os.path.isfile(path):
        return "Không tìm thấy ảnh", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/debug_image/<filename>")
def debug_image(filename):
    """Serve ảnh debug."""
    path = os.path.join(DEBUG_DIR, filename)
    if not os.path.isfile(path):
        return "Không tìm thấy ảnh debug", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/download/json/<filename>")
def download_json(filename):
    """Tải xuống kết quả JSON."""
    path = os.path.join(RESULT_DIR, filename)
    if not os.path.isfile(path):
        return "Không tìm thấy file", 404
    return send_file(path, as_attachment=True, download_name=filename)


@app.route("/download/csv/<result_id>")
def download_csv(result_id):
    """Tải xuống kết quả dạng CSV."""
    json_path = os.path.join(RESULT_DIR, f"results_{result_id}.json")
    if not os.path.isfile(json_path):
        return "Không tìm thấy kết quả", 404

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    fc_cols = [f"FC_{i}" for i in range(1, 41)]
    tf_cols = [f"TF_{i}" for i in range(1, 33)]
    dg_cols = [f"DG_{i}" for i in range(1, 7)]
    writer.writerow(["STT", "File", "SBD", "MĐT"] + fc_cols + tf_cols + dg_cols + ["Cảnh báo", "Lỗi"])

    for idx, item in enumerate(data):
        row = [idx + 1, item.get("_original_name", item.get("org", "")),
               item.get("sbd", ""), item.get("mdt", "")]

        res = item.get("res")
        if res:
            fc = res.get("fc", {})
            for i in range(1, 41):
                ans = fc.get(str(i), [])
                labels = ["A", "B", "C", "D"]
                row.append(",".join(labels[a] for a in ans if a < len(labels)))

            tf = res.get("tf", {})
            for i in range(1, 33):
                ans = tf.get(str(i), [])
                labels = ["Đ", "S"]
                row.append(",".join(labels[a] for a in ans if a < len(labels)))

            dg = res.get("dg", {})
            for i in range(1, 7):
                row.append(dg.get(str(i), ""))
        else:
            row.extend([""] * (40 + 32 + 6))

        row.append(item.get("warn", ""))
        row.append("; ".join(item.get("err", [])))
        writer.writerow(row)

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8-sig")),
        as_attachment=True,
        download_name=f"ketqua_{result_id}.csv",
        mimetype="text/csv"
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
