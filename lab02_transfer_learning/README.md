# Lab 2 - Transfer Learning với mô hình Pretrained (PyTorch)

## Mục tiêu

Thực hành sử dụng các kiến trúc đã huấn luyện trước từ `torchvision.models`, điều chỉnh
chúng cho bài toán phân loại FashionMNIST và so sánh hai chiến lược transfer learning:

- Đóng băng backbone và chỉ huấn luyện lớp phân loại mới.
- Mở khóa một số lớp cuối để fine-tune với learning rate nhỏ hơn.

Lab cũng khảo sát ảnh hưởng của siêu tham số, sử dụng TensorBoard để theo dõi quá trình
huấn luyện, đánh giá trên tập test và lưu đầy đủ artifact để kết quả có thể kiểm chứng.

## Nội dung đã thực hiện

- Sử dụng hai kiến trúc pretrained khác nhau: **ResNet18** và **MobileNetV2**.
- Nạp trọng số `IMAGENET1K_V1` bằng weights API của `torchvision`.
- Thay lớp phân loại ImageNet 1.000 lớp bằng đầu ra 10 lớp FashionMNIST.
- Thử nghiệm đóng băng backbone và fine-tune `layer4` + `fc` của ResNet18.
- Thay đổi learning rate từ `0.001` xuống `0.0001` khi fine-tune.
- So sánh MobileNetV2 với batch size `32` và `64`, giữ các yếu tố khác không đổi.
- Ghi loss, accuracy và learning rate của từng epoch bằng TensorBoard.
- In cấu trúc model, tổng số tham số, số tham số trainable và tên các tensor được cập nhật.
- Đánh giá bằng test accuracy, confusion matrix, chỉ số từng lớp và các mẫu phân loại sai.

## Bộ dữ liệu

[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) gồm 60.000 ảnh train và
10.000 ảnh test thuộc 10 lớp trang phục. Tập train chính thức được chia cố định thành 90%
train và 10% validation với `seed: 42`.

Để phù hợp với backbone pretrained trên ImageNet, ảnh được:

1. Resize từ 28x28 lên 224x224.
2. Chuyển từ ảnh xám một kênh sang ba kênh.
3. Chuẩn hóa bằng mean/std của ImageNet.
4. Áp dụng augmentation cho thí nghiệm fine-tune ResNet18.

Dữ liệu tải về nằm trong `data/` và được bỏ qua bởi Git. Code sẽ tự tải FashionMNIST và
tạo lại train/validation split khi chưa có dữ liệu cục bộ.

## Pipeline

```text
FashionMNIST -> Resize 224x224 -> Grayscale thành 3 kênh -> ImageNet normalization
             -> DataLoader train/validation/test
             -> Nạp model pretrained -> Thay classifier 10 lớp
             -> Freeze hoặc fine-tune -> Train + TensorBoard
             -> Chọn checkpoint theo validation loss thấp nhất
             -> Đánh giá test -> Confusion matrix -> Phân tích lỗi -> So sánh
```

## Cấu trúc dự án

```text
lab02_transfer_learning/
├── configs/            # Cấu hình YAML cho từng thí nghiệm
├── data/               # FashionMNIST và split train/validation cục bộ (gitignored)
├── experiments/        # Bảng tổng hợp kết quả chính thức
├── notebooks/          # So sánh mô hình và phân tích lỗi
├── runs/               # Checkpoint, metrics, plots và TensorBoard của từng run
├── scripts/            # Điểm vào để train, evaluate và tổng hợp kết quả
├── src/                # Dataset, models, training engine, evaluation và visualization
├── requirements.txt
└── README.md
```

## Cài đặt

Yêu cầu Python 3 và các package trong `requirements.txt`.

```powershell
cd lab02_transfer_learning
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Thiết bị được chọn tự động theo thứ tự CUDA, Apple MPS rồi CPU.

## Các cấu hình thí nghiệm

| Config | Kiến trúc | Chiến lược | Batch size | Learning rate |
|---|---|---|---:|---:|
| `resnet18_pretrained.yaml` | ResNet18 | Đóng băng backbone, train `fc` | 32 | 0.001 |
| `resnet18_finetune.yaml` | ResNet18 | Fine-tune `layer4` và `fc` | 32 | 0.0001 |
| `mobilenet_v2_pretrained.yaml` | MobileNetV2 | Đóng băng features, train classifier | 32 | 0.001 |
| `mobilenet_v2_batch64.yaml` | MobileNetV2 | Giữ nguyên baseline, đổi batch size | 64 | 0.001 |

Mỗi cấu hình ghi rõ mục đích, giả thuyết, biến được thay đổi, các yếu tố giữ nguyên và
kết quả kỳ vọng.

## Huấn luyện

Chạy một thí nghiệm bằng config tương ứng:

```powershell
python scripts/train.py --config configs/resnet18_pretrained.yaml
python scripts/train.py --config configs/resnet18_finetune.yaml
python scripts/train.py --config configs/mobilenet_v2_pretrained.yaml
python scripts/train.py --config configs/mobilenet_v2_batch64.yaml
```

Mỗi lệnh tạo một thư mục mới `runs/<model>/run_XXX/`. Run cũ không bị ghi đè. Checkpoint
được lưu từ epoch có validation loss thấp nhất, không mặc định lấy epoch cuối.

## Đánh giá

Ví dụ đánh giá run MobileNetV2 batch size 64:

```powershell
python scripts/evaluate.py --run runs/mobilenet_v2/run_003
```

Lệnh đánh giá nạp `best_model.pt`, chạy trên tập test chính thức và lưu:

- `metrics/final_metrics.json`
- `metrics/confusion_matrix.csv`
- `metrics/per_class_metrics.csv`
- `metrics/misclassified.csv`
- `plots/confusion_matrix.png`

Tái tạo bảng tổng hợp từ tất cả run hoàn chỉnh:

```powershell
python scripts/compare_models.py
```

Kết quả được ghi vào `experiments/experiment_summary.csv`. Cột `conclusion` đã viết thủ
công được giữ nguyên khi chạy lại script.

## TensorBoard

Mỗi run ghi các scalar sau cho từng epoch:

- `Loss/train`
- `Loss/validation`
- `Accuracy/train`
- `Accuracy/validation`
- `LearningRate`

Khởi động TensorBoard từ thư mục Lab 2:

```powershell
tensorboard --logdir runs
```

Sau đó mở địa chỉ được TensorBoard in ra, mặc định là `http://localhost:6006`.

## Notebook

Khởi động Jupyter:

```powershell
python -m jupyter lab
```

Mở notebook theo thứ tự:

1. `notebooks/02_training_and_comparison.ipynb`
2. `notebooks/03_error_analysis.ipynb`

Notebook so sánh chỉ đọc artifact đã lưu, không huấn luyện lại. Nội dung gồm:

- Bảng so sánh cấu hình và kết quả của bốn run.
- Đường cong train/validation loss và accuracy.
- In cấu trúc đầy đủ của MobileNetV2 và ResNet18.
- Liệt kê tham số trainable cho từng chiến lược frozen/fine-tune.
- Kiểm chứng mỗi run có đủ 5 epoch TensorBoard.
- Nhận xét ảnh hưởng của batch size và chiến lược fine-tune.

## Kết quả chính thức

| Run | Thiết lập | Best val accuracy | Test accuracy |
|---|---|---:|---:|
| `mobilenet_v2/run_002` | Frozen, batch 32 | 85.28% | 85.12% |
| `mobilenet_v2/run_003` | Frozen, batch 64 | 85.95% | 85.43% |
| `resnet18/run_001` | Frozen, batch 32 | 86.38% | 86.44% |
| `resnet18/run_003` | Fine-tune `layer4` + `fc` | **93.20%** | **92.87%** |

### Nhận xét

- Tăng batch size MobileNetV2 từ 32 lên 64 làm best validation accuracy tăng 0,67 điểm
  phần trăm và test accuracy tăng 0,31 điểm phần trăm. Cải thiện có thật nhưng nhỏ.
- ResNet18 frozen tốt hơn MobileNetV2 batch 64 khoảng 1,01 điểm test accuracy, đổi lại có
  nhiều tham số hơn.
- Fine-tune `layer4` và `fc` của ResNet18 với learning rate `0.0001` cải thiện test accuracy
  6,43 điểm phần trăm so với ResNet18 frozen.
- `resnet18/run_003` là mô hình tốt nhất và được chọn cho notebook phân tích lỗi.

## Artifact của một run

Một run hoàn chỉnh có cấu trúc:

```text
runs/<model>/run_XXX/
├── config.yaml
├── description.md
├── checkpoints/best_model.pt
├── metrics/
│   ├── history.csv
│   ├── final_metrics.json
│   ├── confusion_matrix.csv
│   ├── per_class_metrics.csv
│   └── misclassified.csv
├── plots/
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   └── confusion_matrix.png
└── tensorboard/events.out.tfevents.*
```

## Khả năng tái lập

- Tất cả thí nghiệm dùng `seed: 42`.
- Các run dùng cùng train/validation split để có thể so sánh trực tiếp.
- Config thực tế được sao chép vào từng run.
- Checkpoint chỉ lưu `state_dict`; script đánh giá dựng lại kiến trúc rồi nạp checkpoint.
- Bảng kết quả được tạo từ `config.yaml` và `metrics/final_metrics.json`, không nhập số liệu
  huấn luyện bằng tay.

## Đối chiếu yêu cầu Practice 2

| Yêu cầu | Phần thực hiện |
|---|---|
| Thử nhiều model pretrained | ResNet18 và MobileNetV2 |
| Điều chỉnh hyperparameter | Learning rate, batch size và augmentation |
| Theo dõi bằng TensorBoard | Năm scalar cho từng epoch của mọi run |
| Khám phá kiến trúc | Notebook in model và tham số trainable |
| Điều chỉnh cho bài toán cụ thể | Thay classifier thành 10 lớp FashionMNIST |
| Freeze/fine-tune | Frozen backbone và fine-tune `layer4` + `fc` |
| Chuẩn bị dữ liệu | `transforms`, ImageNet normalization và `DataLoader` |
| Huấn luyện, đánh giá | CrossEntropyLoss, Adam, validation và test metrics |
