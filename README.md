
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
# Lab 2 - Transfer Learning với PyTorch

Thực hành sử dụng các mô hình pretrained từ `torchvision.models` để phân loại ảnh FashionMNIST.

## Nội dung chính

- Sử dụng ResNet18 và MobileNetV2 pretrained trên ImageNet.
- Thay classifier thành đầu ra 10 lớp FashionMNIST.
- Thử nghiệm frozen backbone và fine-tune các lớp cuối.
- So sánh learning rate và batch size.
- Theo dõi huấn luyện bằng TensorBoard.
- Đánh giá bằng accuracy, confusion matrix và per-class metrics.

## Kết quả

| Mô hình | Thiết lập | Validation | Test |
|---|---|---:|---:|
| MobileNetV2 | Frozen, batch 32 | 85.28% | 85.12% |
| MobileNetV2 | Frozen, batch 64 | 85.95% | 85.43% |
| ResNet18 | Frozen | 86.38% | 86.44% |
| ResNet18 | Fine-tune `layer4` + `fc` | **93.20%** | **92.87%** |

ResNet18 fine-tune đạt kết quả tốt nhất và được chọn để phân tích lỗi.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Huấn luyện

```bash
python scripts/train.py --config configs/resnet18_pretrained.yaml
python scripts/train.py --config configs/resnet18_finetune.yaml
python scripts/train.py --config configs/mobilenet_v2_pretrained.yaml
python scripts/train.py --config configs/mobilenet_v2_batch64.yaml
```

## Đánh giá

```bash
python scripts/evaluate.py --run runs/resnet18/run_003
python scripts/compare_models.py
```

## TensorBoard

```bash
tensorboard --logdir runs
```

Chi tiết quá trình huấn luyện, so sánh mô hình và phân tích lỗi nằm trong thư mục `notebooks/`.
