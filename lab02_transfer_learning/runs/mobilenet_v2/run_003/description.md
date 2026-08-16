# mobilenet_v2_batch64

## Purpose
Đánh giá ảnh hưởng của batch size lớn hơn khi huấn luyện lớp phân loại của MobileNetV2 pretrained trên FashionMNIST.

## Hypothesis
Tăng batch size từ 32 lên 64 sẽ giảm số bước cập nhật mỗi epoch và có thể làm quá trình huấn luyện ổn định hơn, nhưng độ chính xác có thể thay đổi do số lần cập nhật trọng số ít hơn.

## Changed from baseline
Chỉ tăng batch size từ 32 lên 64 so với thí nghiệm MobileNetV2 pretrained; các thành phần còn lại được giữ nguyên để tạo đối chứng công bằng.

## Kept constant
Giữ nguyên MobileNetV2 pretrained, ImageNet weights, backbone đóng băng, learning rate 0.001, Adam, 5 epoch, seed 42, cách chia dữ liệu và tiền xử lý.

## Expected observation
Kết quả validation và test sẽ cho biết batch size 64 có cải thiện độ ổn định hoặc độ chính xác so với baseline batch size 32 hay không.

## Observed result
- Best epoch: 4 (lowest validation loss)
- Best validation loss: 0.3903
- Best validation accuracy: 0.8595
- Final train accuracy: 0.8406
- Final validation accuracy: 0.8385
- Final train/validation accuracy gap: 0.0021

## Interpretation
- Test accuracy: 0.8543.
- So với MobileNetV2 batch size 32, batch size 64 tăng best validation accuracy từ
  0.8528 lên 0.8595 (+0.0067) và test accuracy từ 0.8512 lên 0.8543 (+0.0031).
- Cải thiện là nhỏ; validation loss tăng lại ở epoch 5 nên checkpoint epoch 4 được giữ.
- Xem `notebooks/02_training_and_comparison.ipynb` để đối chiếu với các run khác.
