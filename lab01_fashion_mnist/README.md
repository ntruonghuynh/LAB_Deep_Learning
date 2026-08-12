# Lab 1 - Phan loai FashionMNIST bang PyTorch

Project nay xay dung, huan luyen, danh gia va so sanh cac mo hinh MLP/CNN don gian cho bai toan phan loai anh FashionMNIST. Code duoc viet theo kieu PyTorch thu cong de de hieu pipeline: `Dataset`/`DataLoader`, transforms, training loop, validation, checkpoint, confusion matrix va phan tich loi.

## 1. Cau truc project

```text
lab01_fashion_mnist/
|-- configs/            # File cau hinh YAML cho tung thi nghiem
|-- data/               # Du lieu FashionMNIST tai ve va split train/validation co dinh
|-- experiments/        # Bang tong hop ket qua cac run
|-- notebooks/          # Notebook kham pha, so sanh, phan tich loi
|-- report/figures/     # Hinh anh dung cho bao cao
|-- runs/               # Ket qua train/evaluate, checkpoint, metrics, plots
|-- scripts/            # Lenh chay train/evaluate/compare
|-- src/                # Source code chinh
|-- requirements.txt    # Danh sach thu vien can cai
+-- README.md
```

Thu muc project dung de chay lenh la:

```text
E:\Deep_learning\LAB1_Deep_Learning\lab01_fashion_mnist
```

Neu ban dang o:

```text
E:\Deep_learning\LAB1_Deep_Learning
```

thi can `cd lab01_fashion_mnist` truoc khi cai thu vien hoac chay code.

## 2. Cai dat tren Windows

Mo terminal PowerShell hoac terminal trong VS Code, chay:

```powershell
cd E:\Deep_learning\LAB1_Deep_Learning\lab01_fashion_mnist
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Neu PowerShell chan activate script, chay lenh nay mot lan:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Sau do activate lai:

```powershell
.\.venv\Scripts\Activate.ps1
```

Kiem tra nhanh thu vien:

```powershell
python -c "import torch, torchvision, pandas, yaml, sklearn; print('OK')"
```

## 3. Cai dat tren macOS/Linux

```bash
cd /duong/dan/toi/lab01_fashion_mnist
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kiem tra nhanh thu vien:

```bash
python -c "import torch, torchvision, pandas, yaml, sklearn; print('OK')"
```

## 4. Cach train model

Chay trong thu muc `lab01_fashion_mnist`.

Train MLP baseline:

```powershell
python scripts\train.py --config configs\mlp_baseline.yaml
```

Train CNN baseline:

```powershell
python scripts\train.py --config configs\cnn_baseline.yaml
```

Train CNN experiment:

```powershell
python scripts\train.py --config configs\cnn_experiment.yaml
```

Tren macOS/Linux co the dung dau `/` thay vi `\`:

```bash
python scripts/train.py --config configs/mlp_baseline.yaml
python scripts/train.py --config configs/cnn_baseline.yaml
python scripts/train.py --config configs/cnn_experiment.yaml
```

Moi lan train se tao mot thu muc run moi:

```text
runs/<model>/run_XXX/
```

Vi du:

```text
runs/mlp/run_001/
runs/cnn/run_002/
```

Ben trong moi run co:

```text
config.yaml
description.md
metrics/history.csv
metrics/final_metrics.json
plots/loss_curve.png
plots/accuracy_curve.png
checkpoints/best_model.pt
```

## 5. Cach evaluate model

Evaluate checkpoint tot nhat cua mot run tren test set:

```powershell
python scripts\evaluate.py --run runs\cnn\run_002
```

macOS/Linux:

```bash
python scripts/evaluate.py --run runs/cnn/run_002
```

Sau khi evaluate, project se luu them:

```text
metrics/confusion_matrix.csv
metrics/per_class_metrics.csv
metrics/misclassified.csv
plots/confusion_matrix.png
```

Dong thoi `metrics/final_metrics.json` se duoc cap nhat them `test_accuracy`.

## 6. So sanh cac model

Tao lai bang tong hop ket qua:

```powershell
python scripts\compare_models.py
```

Ket qua duoc luu tai:

```text
experiments/experiment_summary.csv
```

## 7. Xu ly loi tai FashionMNIST

Lan chay dau tien se tu dong tai FashionMNIST vao:

```text
data/FashionMNIST/raw
```

Neu mang bi timeout hoac bi chan, ban co the thay loi dang:

```text
RuntimeError: Error downloading ...
Connection attempt failed
No connection could be made because the target machine actively refused it
```

Project da cau hinh them mirror GitHub chinh thuc trong `src/dataset.py`, nhung neu mang van loi thi tai thu cong 4 file sau:

```text
https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz
https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz
https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz
https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz
```

Dat ca 4 file vao:

```text
E:\Deep_learning\LAB1_Deep_Learning\lab01_fashion_mnist\data\FashionMNIST\raw
```

Sau do chay lai lenh train.

## 8. Loi encoding tren Windows

Project co cac file YAML va mo ta thi nghiem bang tieng Viet UTF-8. Tren Windows, neu code khong chi dinh encoding co the gap loi:

```text
UnicodeDecodeError: 'charmap' codec can't decode byte ...
UnicodeEncodeError: 'charmap' codec can't encode character ...
```

Code hien tai da duoc cap nhat de doc/ghi UTF-8 trong cac file:

```text
scripts/train.py
scripts/evaluate.py
scripts/compare_models.py
src/dataset.py
src/run_manager.py
```

Vi vay neu dang dung ban moi nhat, ban chi can chay lai lenh train.

## 9. Thiet bi chay

Project tu dong chon thiet bi tot nhat co san:

```text
CUDA > Apple MPS > CPU
```

- Windows co GPU NVIDIA va cai PyTorch CUDA dung cach: chay bang `cuda`.
- Mac Apple Silicon: co the chay bang `mps`.
- Neu khong co GPU phu hop: chay bang `cpu`.

Tren may Windows hien tai, log co the hien:

```text
Su dung thiet bi: cpu
```

Dieu nay binh thuong, chi la train se cham hon GPU.

## 10. Cac model trong project

`FashionMLP` trong `src/models.py`:

```text
Flatten -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear(10)
```

`FashionCNN` trong `src/models.py`:

```text
Conv2d -> ReLU -> MaxPool
Conv2d -> ReLU -> MaxPool
Flatten -> Linear(128) -> ReLU -> Dropout -> Linear(10)
```

## 11. Ket qua hien co

| Run | Model | Best validation accuracy | Test accuracy |
|---|---|---:|---:|
| `mlp/run_001` | MLP baseline | 0.8902 | 0.8847 |
| `cnn/run_001` | CNN baseline, dropout 0.3 | 0.9195 | 0.9161 |
| `cnn/run_002` | CNN experiment, dropout 0.5 | 0.9262 | 0.9204 |

Model tot nhat hien tai la:

```text
runs/cnn/run_002
```

## 12. Notebook

Mo cac notebook theo thu tu:

```text
notebooks/01_data_exploration.ipynb
notebooks/02_training_and_comparison.ipynb
notebooks/03_error_analysis.ipynb
```

Chon kernel la moi truong `.venv` cua project.

## 13. Tom tat pipeline

```text
Tai FashionMNIST
-> Chuan hoa anh
-> Chia train/validation bang split co dinh
-> Train MLP/CNN
-> Theo doi train loss, validation loss, accuracy
-> Luu checkpoint tot nhat theo validation loss
-> Evaluate tren test set
-> Luu confusion matrix, per-class metrics, misclassified samples
-> So sanh model va ket luan
```
