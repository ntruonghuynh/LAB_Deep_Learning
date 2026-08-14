# Lab 1 - FashionMNIST Classification with PyTorch

Dự án này xây dựng, huấn luyện, đánh giá và so sánh các mô hình phân loại ảnh FashionMNIST bằng PyTorch. Mục tiêu chính là hiểu toàn bộ pipeline học sâu cơ bản: chuẩn bị dữ liệu, xây dựng mô hình, huấn luyện, đánh giá, lưu kết quả và phân tích lỗi.

## Mục Tiêu

- Phân loại ảnh FashionMNIST gồm 10 lớp trang phục.
- So sánh mô hình MLP baseline và CNN.
- Theo dõi loss/accuracy trong quá trình huấn luyện.
- Đánh giá mô hình trên test set chính thức.
- Lưu checkpoint, metrics, biểu đồ và confusion matrix.
- Phân tích các mẫu bị phân loại sai.

## Cấu Trúc Dự Án

```text
lab01_fashion_mnist/
|-- configs/            # File cấu hình YAML cho từng thí nghiệm
|-- data/               # Dữ liệu FashionMNIST và split train/validation
|-- experiments/        # Bảng tổng hợp kết quả các lần chạy
|-- notebooks/          # Notebook khám phá dữ liệu, so sánh, phân tích lỗi
|-- report/figures/     # Hình ảnh dùng cho báo cáo
|-- runs/               # Kết quả train/evaluate, checkpoint, metrics, plots
|-- scripts/            # Các file chạy train, evaluate, compare
|-- src/                # Source code chính
|-- requirements.txt    # Danh sách thư viện cần cài
+-- README.md
