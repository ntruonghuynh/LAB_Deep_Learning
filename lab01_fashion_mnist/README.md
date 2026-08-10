# Lab 1 - Phân loại FashionMNIST (PyTorch)

## Mục tiêu

Xây dựng, huấn luyện và so sánh các bộ phân loại MLP và CNN đơn giản trên FashionMNIST bằng
các khối xây dựng cốt lõi của PyTorch (`Dataset`/`DataLoader`, `transforms`, vòng lặp huấn
luyện thủ công), đánh giá chúng đúng cách (không chỉ dựa vào accuracy cuối cùng), và thực
hiện phân tích lỗi thực sự dựa trên các kết quả đã lưu thật.

## Bộ dữ liệu

[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) thông qua
`torchvision.datasets`: 10 lớp ảnh trang phục xám 28x28, 60.000 ảnh huấn luyện chính thức
và 10.000 ảnh kiểm tra chính thức, cân bằng hoàn hảo giữa các lớp.

## Pipeline Machine Learning

```
Bài toán -> Khám phá dữ liệu -> Tiền xử lý/Biến đổi -> Chia tập Train/Val/Test
        -> Mô hình cơ sở -> Huấn luyện -> Giám sát -> Validation/Lựa chọn mô hình
        -> Thí nghiệm -> Đánh giá trên tập test cuối cùng -> Ma trận nhầm lẫn
        -> Phân tích lỗi phân loại sai -> So sánh mô hình -> Kết luận
```

Đây là luồng mà các notebook (và bất kỳ phần trình bày nào của lab này) nên tuân theo -
không phải là đi qua từng file một cách rời rạc.

## Cấu trúc dự án

```
lab01_fashion_mnist/
├── configs/            # Config YAML, mỗi file cho một thí nghiệm chính thức
├── notebooks/          # khám phá / giám sát / so sánh / phân tích lỗi (lớp trình bày)
├── src/                # mã nguồn tái sử dụng: dataset, models, engine, evaluation, metrics, visualization, run_manager
├── scripts/             # train.py, evaluate.py, compare_models.py (điểm vào dòng lệnh)
├── data/                # bộ nhớ đệm FashionMNIST (gitignored) + split train/val cố định (đã commit)
├── experiments/         # experiment_summary.csv (tổng hợp, chỉ chứa kết quả thật)
├── runs/                # một thư mục cho mỗi lần chạy chính thức: config, description, metrics, plots, checkpoint
└── report/figures/      # các hình được chọn cho báo cáo/trình bày cuối cùng
```

## Cài đặt

```bash
cd lab01_fashion_mnist
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tự động chạy trên CPU, CUDA, hoặc Apple MPS (`src/dataset.get_device()` chọn thiết bị tốt
nhất hiện có; mọi thí nghiệm trong repo này được chạy trên MPS).

## Cách huấn luyện

```bash
python scripts/train.py --config configs/mlp_baseline.yaml
python scripts/train.py --config configs/cnn_baseline.yaml
python scripts/train.py --config configs/cnn_experiment.yaml
```

Mỗi lệnh gọi tạo ra một thư mục run mới dưới `runs/<model>/run_XXX/` (tự động tăng số,
không bao giờ bị ghi đè) chứa `config.yaml`, `description.md`, `metrics/history.csv`,
`metrics/final_metrics.json`, `plots/loss_curve.png`, `plots/accuracy_curve.png`, và
`checkpoints/best_model.pt` (state_dict của epoch có validation loss thấp nhất).

## Cách đánh giá

```bash
python scripts/evaluate.py --run runs/cnn/run_002
```

Nạp `config.yaml` + checkpoint tốt nhất của lần chạy đó, đánh giá một lần duy nhất trên
**tập test chính thức**, và lưu `metrics/confusion_matrix.csv`, `metrics/per_class_metrics.csv`,
`metrics/misclassified.csv`, và `plots/confusion_matrix.png` vào cùng thư mục run. Nó cũng
thêm `test_accuracy` vào `final_metrics.json` của lần chạy đó.

Để tái tạo bảng so sánh giữa các lần chạy:

```bash
python scripts/compare_models.py
```

Quét qua mọi lần chạy chính thức (bất kỳ thư mục nào có cả `config.yaml` và
`metrics/final_metrics.json`) và ghi lại `experiments/experiment_summary.csv` - giữ nguyên
bất kỳ văn bản `conclusion` nào đã được viết cho một lần chạy.

## Cách chạy các Notebook

Mở `notebooks/01_data_exploration.ipynb`, `02_training_and_comparison.ipynb`, và
`03_error_analysis.ipynb` theo thứ tự, sử dụng `.venv` của dự án làm kernel. Cả ba đều chạy
từ đầu đến cuối mà không huấn luyện lại gì cả - chúng chỉ nạp các artifact đã được lưu sẵn
dưới `runs/`, `data/splits/`, và `experiments/`.

## Quy ước thí nghiệm

- Mọi lần huấn luyện có ý nghĩa đều được khởi chạy qua `scripts/train.py` với một config từ
  `configs/`, tạo ra một lần chạy chính thức dưới `runs/<model>/run_XXX/`.
- Các lần chạy debug/kiểm tra nhanh (ví dụ: kiểm tra nhanh 1 epoch trong lúc phát triển) được
  thực hiện với các config tạm thời bên ngoài `configs/` và bị xóa sau đó - chúng không bao
  giờ trở thành lần chạy chính thức.
- Các lần chạy chính thức không bao giờ bị ghi đè; ID của lần chạy tiếp theo được tính tự động.
- Việc lựa chọn mô hình giữa các thí nghiệm chỉ dựa trên số liệu **validation**. Tập test
  chính thức chỉ được dùng đúng một lần cho mỗi lần chạy được chọn, thuần túy để báo cáo cuối
  cùng.

### Các lần chạy chính thức trong repo này

| Lần chạy | Mục đích | Val acc tốt nhất | Test acc |
|---|---|---|---|
| `mlp/run_001` | MLP cơ sở | 0.8902 | 0.8847 |
| `cnn/run_001` | CNN cơ sở | 0.9195 | 0.9161 |
| `cnn/run_002` | Thí nghiệm CNN: dropout 0.3 -> 0.5 | 0.9262 | 0.9204 |

`cnn/run_002` có validation accuracy tốt nhất và khoảng cách train/validation nhỏ nhất, vì
vậy đây là mô hình được dùng trong notebook phân tích lỗi. Xem `description.md` của mỗi lần
chạy để biết đầy đủ mục đích/giả thuyết/kết quả/kết luận, và
`experiments/experiment_summary.csv` cho bảng so sánh dạng máy có thể đọc được.

## Nơi lưu Runs/Kết quả

- `runs/<model>/run_XXX/` - mọi thứ về một lần chạy chính thức (config, description,
  metrics, plots, checkpoint).
- `experiments/experiment_summary.csv` - một dòng cho mỗi lần chạy chính thức, để so sánh
  nhanh.
- `data/splits/split_seed42.json` - split chỉ số train/validation cố định, dùng chung bởi
  mọi lần chạy với `seed: 42`.

## Các mô hình chính

- **`FashionMLP`** (`src/models.py`): Flatten -> [Linear -> ReLU -> Dropout] x2 -> Linear(10).
- **`FashionCNN`** (`src/models.py`): hai khối Conv(3x3)-ReLU-MaxPool (kênh 1->32->64) theo
  sau bởi một đầu phân loại Linear(128) -> Dropout -> Linear(10).

## Ghi chú về khả năng tái lập

- `src/seed.py` đặt seed cho `random` của Python, NumPy, và PyTorch (CPU + CUDA/MPS) chỉ
  bằng một lệnh gọi; seed được sử dụng luôn được ghi lại trong `config.yaml` của lần chạy.
- Mọi thí nghiệm trong repo này dùng `seed: 42` và cùng một split đã lưu
  (`data/splits/split_seed42.json`), vì vậy các lần chạy MLP và CNN được huấn luyện/kiểm
  định trên cùng một dữ liệu và có thể so sánh trực tiếp với nhau.
- Checkpoint của mô hình chỉ lưu `state_dict` (không lưu toàn bộ mô hình đã pickle);
  `scripts/evaluate.py` dựng lại kiến trúc từ `config.yaml` và nạp state_dict, đây chính là
  cách mọi test accuracy được báo cáo ở trên thực sự thu được (không có số liệu bịa đặt).
