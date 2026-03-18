# Tổng quan dự án `duan2` (đọc từ gốc đến ngọn)

## 1) Mục tiêu dự án

Dự án xây dựng hệ thống chấm phiếu trắc nghiệm THPT 2025 bằng xử lý ảnh:

- Nhận ảnh phiếu (scan/chụp)
- Phát hiện biên phiếu và chuẩn hóa phối cảnh
- Trích xuất 5 phần dữ liệu:
  - `sbd` (Số báo danh, 6 chữ số)
  - `mdt` (Mã đề thi, 3 chữ số)
  - `fc` (40 câu nhiều lựa chọn A/B/C/D)
  - `tf` (32 ô đúng/sai theo 8 nhóm × 4 ý)
  - `dg` (6 câu trả lời ngắn dạng lưới ký tự)
- Xuất kết quả qua:
  - Web UI (ảnh đánh dấu + bảng kết quả)
  - API JSON
  - CSV tải về
  - Batch CLI

---

## 2) Cấu trúc mã nguồn và dữ liệu

### 2.1 Thư mục/chức năng chính

- `app.py`: Flask app, route web + API + tải JSON/CSV
- `process_batch.py`: xử lý hàng loạt thư mục ảnh
- `config.json`: toàn bộ tham số pipeline và tọa độ vùng
- `core/`: engine xử lý chính
  - `preprocessor.py`: tiền xử lý ảnh
  - `detector.py`: phát hiện phiếu, warp, deskew, align template
  - `extractor.py`: đọc đáp án theo lưới bubble
  - `visualizer.py`: vẽ overlay kết quả
  - `__init__.py`: pipeline orchestration (process_image, process_batch)
- `templates/`, `static/css/`: giao diện web
- `anh/`: dữ liệu mẫu + ảnh test + ảnh debug tạo ra trong quá trình thử nghiệm
- `uploads/`, `results/`, `debug/`: dữ liệu runtime web

### 2.2 Quy mô hiện tại (theo nội dung repo)

- Python files: 29 (`24` file ở root + `5` file trong `core/`)
- Test/diagnostic scripts: 14 file `test_*.py`
- Dữ liệu ảnh test:
  - `anh/test_imga`: 154 file
  - `anh/sample_image`: 5 file

---

## 3) Luồng xử lý end-to-end

### 3.1 Luồng web (`app.py`)

1. Người dùng upload 1 hoặc nhiều ảnh ở route `/process`
2. Mỗi ảnh được lưu vào `uploads/` với UUID prefix
3. Gọi `core.process_image(...)`
4. Nếu thành công:
   - lưu ảnh đánh dấu vào `results/`
   - lưu ảnh debug vào `debug/` (nếu bật debug)
5. Gộp toàn bộ kết quả thành `results_<id>.json`
6. Render `templates/result.html`

Ngoài ra:

- `POST /api/process`: nhận 1 file, trả JSON
- `GET /download/json/<filename>`: tải JSON
- `GET /download/csv/<result_id>`: chuyển JSON sang CSV để tải

### 3.2 Luồng CLI batch (`process_batch.py`)

- Quét ảnh theo extension trong thư mục đầu vào
- Gọi `core.process_batch(image_paths, config, output_dir, debug)`
- Lưu:
  - ảnh đã đánh dấu
  - ảnh debug (nếu cần)
  - `results.json` tổng hợp

### 3.3 Luồng lõi (`core.__init__.process_image`)

Pipeline chuẩn:

1. `load_image` (hỗ trợ đường dẫn Unicode)
2. Tiền xử lý:
   - `preprocess` (gray → CLAHE → blur)
   - `preprocess_to_binary`
3. Tìm góc phiếu (`find_corners`) theo chuỗi fallback:
   - marker corners
   - ink-based
   - contour
4. Warp về kích thước chuẩn (`1800 × 2500` mặc định)
5. Deskew bằng Hough lines
6. Align với template bằng ORB feature matching
7. Tạo binary Otsu riêng cho extraction
8. `extract_all` đọc SBD/MDT/FC/TF/DG
9. `visualize_results` để vẽ overlay
10. Đóng gói output có metadata:
    - `_detection_method`
    - `_skew_angle`

---

## 4) Phân tích từng module `core`

## 4.1 `preprocessor.py`

- `load_image(path)`:
  - dùng `np.fromfile` + `cv2.imdecode` để đọc path Unicode
- Chuỗi transform:
  - grayscale
  - CLAHE (ổn định tương phản với ảnh ánh sáng không đều)
  - Gaussian blur
  - adaptive threshold (binary)
  - Canny (khi cần edge)

## 4.2 `detector.py`

### A. Phát hiện tờ phiếu

- `find_corner_markers`:
  - dò các blob vuông ở rìa ảnh
  - lọc theo area/aspect/solidity
  - gán về 4 góc tham chiếu
- `find_sheet_by_ink`:
  - threshold Otsu đảo
  - morphological close
  - lấy toàn bộ pixel mực, fit `minAreaRect`
- `find_sheet_contour`:
  - fallback cuối, tìm contour tứ giác lớn nhất

`find_corners` dùng thứ tự ưu tiên:
`markers` → `ink` → `contour`

### B. Chuẩn hóa hình học

- `warp_perspective`: đưa về khung chuẩn
- `deskew`:
  - HoughLinesP
  - tính góc nghiêng trung bình có trọng số chiều dài line
  - bỏ qua nếu góc quá nhỏ (<0.3°) hoặc quá lớn (>15°)

### C. Căn chỉnh theo template

- `TemplateAligner` (singleton):
  - load + warp template 1 lần
  - ORB keypoints + BFMatcher (knn ratio test)
  - RANSAC homography để align ảnh đã warp

## 4.3 `extractor.py`

Đây là phần “nặng” nhất và là giá trị chính của hệ thống.

### A. Tọa độ bubble

- `get_bubble_rect`: map tọa độ `%` vùng sang pixel
- `compute_fill_ratio`: tỉ lệ pixel “được tô” trong ô

### B. Robust detection cho edge cases

Hàm trọng tâm: `robust_bubble_detection(...)` (multi-stage)

- Stage 1: quyết định nhanh theo `fill_threshold`
- Stage 2: lọc noise bằng hình dạng (`is_valid_mark`)
- Stage 3: phát hiện vết tẩy/xóa (`detect_erased_mark`)
- Stage 4: phát hiện tô bút chì nhạt (`detect_pencil_mark`)
- Stage 5: voting nhiều threshold (Otsu/fixed/adaptive)

Edge case đã được encode rõ:

- Case 8: tô rồi xóa
- Case 9: bút chì nhạt
- Case 10: vết bẩn/noise

### C. Trích xuất từng nhóm

- `extract_sbd`, `extract_mdt`:
  - có auto-calibrate trục Y (`auto_calibrate_grid_y`)
  - ưu tiên intensity-based (mean grayscale)
- `extract_fc`:
  - 40 câu, nhóm theo 4 block
  - cảnh báo tô nhiều đáp án hoặc nghi ngờ mờ
- `extract_tf`:
  - logic chính dựa intensity tương đối giữa 2 cột Đ/S
  - fallback robust detection khi thiếu grayscale
- `extract_dg`:
  - đọc theo grid ký tự, không OCR
  - nhãn: `- , 0 1 2 3 4 5 6 7 8 9`

## 4.4 `visualizer.py`

- Vẽ overlay vòng tròn/khung cho từng bubble
- Màu:
  - xanh: tô hợp lệ
  - đỏ: lỗi (ví dụ tô nhiều)
  - xám: trống
- Hiển thị SBD/MDT lên góc ảnh
- `create_debug_image` để vẽ khung toàn bộ vùng config

---

## 5) Cấu hình `config.json`

`config.json` là “trục sống” của hệ thống, gồm:

- Kích thước warp chuẩn (`warp_width`, `warp_height`)
- Đường dẫn template (`template_path`)
- Tham số preprocessing
- Tham số phát hiện marker
- Ngưỡng tô:
  - `fill_threshold` (đang là `0.3`)
  - `double_fill_threshold` (đang là `0.27`)
- `bubble_margin`, `tf_bubble_margin`
- Tọa độ vùng theo tỷ lệ `%` cho:
  - `sbd`, `mdt`
  - `fc.groups` (4 nhóm × 10 hàng × 4 cột)
  - `tf.groups` (8 nhóm × 4 hàng × 2 cột)
  - `dg.groups` (6 câu, mỗi câu 4 cột, 12 hàng ký tự)

---

## 6) Giao diện web

### `templates/index.html`

- Kéo-thả / chọn file
- Upload nhiều file
- Tùy chọn debug mode

### `templates/result.html`

- Card theo từng phiếu:
  - cảnh báo/lỗi
  - ảnh đã đánh dấu
  - chi tiết FC/TF/DG (toggle)
  - ảnh debug (nếu có)
- Nút tải JSON/CSV

### `static/css/style.css`

- UI responsive, card-based
- nhóm màu theo alert/result
- fullscreen overlay cho ảnh

---

## 7) Script test và script phân tích

Dự án có nhiều script test/diagnostic (14 file), phần lớn phục vụ:

- calibrate tọa độ bubble
- so sánh phương pháp detection (`markers` vs `ink`)
- debug line/grid/border
- template matching và align
- kiểm tra fill ratio theo từng vùng

Điểm cần lưu ý:

- Đây chủ yếu là script thủ công (in log + save ảnh), chưa phải test unit tự động theo framework như `pytest`.

---

## 8) Định dạng output hiện tại

Output chuẩn từ pipeline:

```json
{
  "org": "đường dẫn ảnh gốc",
  "out": "đường dẫn/filename ảnh kết quả",
  "warn": "chuỗi cảnh báo",
  "err": ["danh sách lỗi"],
  "res": {
    "fc": {"1": [0], "...": []},
    "tf": {"1": [1], "...": []},
    "dg": {"1": "-0,2", "...": ""}
  },
  "sbd": "000020",
  "mdt": "014"
}
```

Mapping:

- `fc`: `0=A`, `1=B`, `2=C`, `3=D`
- `tf`: `0=Đúng`, `1=Sai`

---

## 9) Nhận định kỹ thuật (thực tế code hiện tại)

### Điểm mạnh

- Pipeline rõ ràng, tách module hợp lý
- Có nhiều tầng fallback cho detection
- Có xử lý edge case thực tế (xóa, bút chì nhạt, noise)
- Có web + API + batch CLI
- Có nhiều script debug/calibration để vận hành thực tế

### Điểm cần cải thiện

1. **Tên file output bị lặp đuôi**:
   - hiện tại tạo dạng `...jpg.jpg` (do ghép thêm `.jpg` trên tên đã có extension)
2. **README hiển thị lỗi dấu tiếng Việt trên một số terminal/codepage**:
   - nội dung file là UTF-8 nhưng console Windows cp1252 dễ bị lỗi
3. **Thiếu test tự động chuẩn hóa**:
   - chưa có bộ assert regression cho chất lượng OCR-like extraction
4. **Nhiều script phân tích nằm ở root**:
   - nên gom vào `tools/` hoặc `scripts/diagnostics/` để dễ bảo trì

---

## 10) Cách chạy nhanh (đúng theo mã nguồn hiện tại)

1. Cài thư viện:

```bash
pip install -r requirements.txt
```

2. Chạy web:

```bash
python app.py
```

3. Batch:

```bash
python process_batch.py anh/test_imga --output results --debug
```

4. Calibrate thủ công:

```bash
python calibrate.py "anh/sample_image/Copy of PhieuTracNghiepTHPT2025.png"
```

---

## 11) Kết luận ngắn

Đây là một dự án chấm phiếu theo hướng **xử lý ảnh cổ điển + rule-based extraction**, được đầu tư khá nhiều vào độ bền thực tế qua cơ chế fallback và script calibrate/diagnostic. Điểm cốt lõi nằm ở `core/extractor.py` (robust bubble detection) và `config.json` (tọa độ vùng). Muốn nâng độ tin cậy dài hạn, ưu tiên tiếp theo nên là: **chuẩn hóa test tự động + quản lý config/versioning cho từng mẫu phiếu**.
