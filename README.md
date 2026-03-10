# Hệ thống chấm phiếu trắc nghiệm THPT 2025

Hệ thống tự động nhận diện và trích xuất đáp án từ phiếu trả lời trắc nghiệm theo mẫu thử nghiệm kỳ thi THPT 2025.

## Tính năng

- **Nhận diện phiếu**: Phát hiện phiếu từ ảnh, xử lý nghiêng/xoay, warp perspective
- **Trích xuất đáp án**: 3 phần (Nhiều lựa chọn, Đúng/Sai, Trả lời ngắn)
- **Xử lý lỗi**: Cảnh báo tô nhiều đáp án, bỏ trống, phiếu hỏng
- **Giao diện web**: Upload, xem kết quả, tải JSON/CSV
- **Xử lý hàng loạt**: Chấm nhiều phiếu cùng lúc

## Cài đặt

### Yêu cầu
- Python 3.8+
- pip

### Bước cài đặt

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Hiệu chỉnh tọa độ (QUAN TRỌNG - làm 1 lần)
python calibrate.py anh/sample_image/Copy\ of\ PhieuTracNghiepTHPT2025.png

# 3. Chạy ứng dụng web
python app.py
```

Truy cập: http://localhost:5000

## Cách sử dụng

### 1. Giao diện Web
- Mở trình duyệt → http://localhost:5000
- Kéo thả hoặc chọn file ảnh phiếu trắc nghiệm
- Bấm "Bắt đầu chấm phiếu"
- Xem kết quả → Tải JSON hoặc CSV

### 2. Xử lý hàng loạt (dòng lệnh)
```bash
python process_batch.py anh/test_imga --output results

# Bật debug để xem chi tiết
python process_batch.py anh/test_imga --output results --debug
```

### 3. API
```bash
curl -X POST -F "file=@phieu.jpg" http://localhost:5000/api/process
```

## Hiệu chỉnh tọa độ (Calibrate)

**Đây là bước quan trọng nhất!** Tọa độ mặc định trong `config.json` là giá trị ước lượng. Cần hiệu chỉnh cho đúng với mẫu phiếu thực tế.

### Cách hiệu chỉnh:

```bash
python calibrate.py anh/sample_image/Copy\ of\ PhieuTracNghiepTHPT2025.png
```

- **Click chuột trái**: In tọa độ % ra console
- **Phím 'd'**: Bật/tắt grid debug
- **Phím 's'**: Lưu ảnh debug
- **Phím 'q'**: Thoát

**Quy trình:**
1. Click vào góc **trên-trái** của vùng SBD → ghi lại (x1, y1)
2. Click vào góc **dưới-phải** của vùng SBD → ghi lại (x2, y2)
3. Cập nhật vào `config.json` → `regions.sbd.x1, y1, x2, y2`
4. Lặp lại cho các vùng: MĐT, FC, TF, DG
5. Chạy lại calibrate.py để kiểm tra grid có khớp không

## Cấu trúc dự án

```
duan2/
├── app.py                  # Flask web application
├── config.json             # Cấu hình tọa độ vùng (editable)
├── requirements.txt        # Dependencies
├── calibrate.py            # Công cụ hiệu chỉnh
├── process_batch.py        # Xử lý hàng loạt
│
├── core/                   # Module xử lý chính
│   ├── __init__.py         # Pipeline chính
│   ├── preprocessor.py     # Tiền xử lý ảnh
│   ├── detector.py         # Phát hiện phiếu + Warp
│   ├── extractor.py        # Trích xuất đáp án
│   └── visualizer.py       # Vẽ kết quả
│
├── templates/              # HTML templates
│   ├── index.html          # Trang upload
│   └── result.html         # Trang kết quả
│
├── static/css/style.css    # CSS styling
│
├── anh/                    # Ảnh mẫu & test
│   ├── output.json         # Mẫu output
│   ├── sample_image/       # Ảnh mẫu phiếu
│   └── test_imga/          # Ảnh test (151 phiếu)
│
├── uploads/                # Ảnh upload (tự tạo)
└── results/                # Kết quả (tự tạo)
```

## Kiến trúc hệ thống

```
Input Image
     │
     ▼
┌──────────────────────┐
│  TIỀN XỬ LÝ          │   Grayscale → CLAHE → GaussianBlur
│  (preprocessor.py)    │   → AdaptiveThreshold / Canny
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  PHÁT HIỆN PHIẾU     │   Corner Markers / Contour Detection
│  (detector.py)        │   → Warp Perspective → Ảnh phẳng chuẩn
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  TRÍCH XUẤT ĐÁP ÁN  │   Grid-based bubble detection
│  (extractor.py)       │   → Fill ratio → Ngưỡng → Đáp án
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  XUẤT KẾT QUẢ        │   JSON + Ảnh đánh dấu + CSV
│  (visualizer.py)      │   + Cảnh báo + Lỗi
└──────────────────────┘
```

## Giải thích thuật toán

### 1. Tiền xử lý
- **CLAHE**: Cân bằng tương phản cục bộ, xử lý ảnh chụp với ánh sáng không đều
- **GaussianBlur**: Khử nhiễu, loại bỏ vết bẩn nhỏ
- **AdaptiveThreshold**: Ngưỡng hóa thích ứng, chuyển ảnh sang đen-trắng
- **Canny**: Phát hiện cạnh, hỗ trợ tìm contour

### 2. Phát hiện phiếu
- **Corner Markers**: Tìm 4 hình vuông đen ở 4 góc phiếu
- **Fallback**: Tìm contour hình chữ nhật lớn nhất
- **Warp Perspective**: Biến đổi phối cảnh → ảnh phẳng kích thước chuẩn

### 3. Nhận diện đáp án (Bubble Detection)
- Sau khi warp, tọa độ bubble là **cố định** (dùng tỉ lệ %)
- `fill_ratio = countNonZero(bubble) / total_pixels`
- `fill_ratio > 40%` → bubble đã được tô
- **Phần III** (trả lời ngắn): Dùng grid bubble (0-9, dấu phẩy, dấu trừ), **KHÔNG dùng OCR**

### 4. Xử lý lỗi
- **Double fill**: 2 ô cùng câu có `fill_ratio > 30%` → Cảnh báo
- **Empty**: Không ô nào đủ ngưỡng → Đánh dấu bỏ trống
- **Phiếu hỏng**: Không tìm thấy corner markers → Thử fallback contour

## Cấu hình (config.json)

| Tham số                 | Mô tả                                    | Mặc định |
| ----------------------- | ---------------------------------------- | -------- |
| `warp_width`            | Chiều rộng ảnh sau warp                  | 1800     |
| `warp_height`           | Chiều cao ảnh sau warp                   | 2500     |
| `fill_threshold`        | Ngưỡng tô đáp án (%)                     | 0.40     |
| `double_fill_threshold` | Ngưỡng cảnh báo tô 2 đáp án              | 0.30     |
| `bubble_margin`         | Margin bên trong mỗi bubble (%)          | 0.15     |
| `regions.*`             | Tọa độ % các vùng (SBD, MĐT, FC, TF, DG) | Xem file |

## Xử lý lỗi thường gặp

| Lỗi                           | Nguyên nhân                  | Cách xử lý                               |
| ----------------------------- | ---------------------------- | ---------------------------------------- |
| Ảnh không load được           | Đường dẫn sai hoặc file hỏng | Kiểm tra đường dẫn, định dạng file       |
| Không tìm thấy corner markers | Ảnh bị cắt/không có marker   | Hệ thống tự fallback sang contour        |
| Kết quả sai                   | Tọa độ config chưa đúng      | Chạy calibrate.py để hiệu chỉnh          |
| Fill ratio không chính xác    | Ngưỡng chưa phù hợp          | Điều chỉnh `fill_threshold` trong config |
| Ảnh quá tối/sáng              | Chất lượng ảnh kém           | CLAHE tự xử lý, nếu quá tệ cần chụp lại  |

## Định dạng output

```json
{
    "org": "đường_dẫn_gốc",
    "out": "đường_dẫn_ảnh_kết_quả",
    "warn": "cảnh báo nếu có",
    "err": [],
    "res": {
        "fc": {"1": [0], "2": [1], ...},
        "tf": {"1": [0], "2": [1], ...},
        "dg": {"1": "-0,0", "2": "0,2", ...}
    },
    "sbd": "012349",
    "mdt": "765"
}
```

- `fc`: 0=A, 1=B, 2=C, 3=D
- `tf`: 0=Đúng, 1=Sai
- `dg`: Chuỗi ký tự (0-9, dấu phẩy, dấu trừ)
