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

Chi tiết quá trình huấn luyện, so sánh mô hình và phân tích lỗi nằm trong thư mục `notebooks/