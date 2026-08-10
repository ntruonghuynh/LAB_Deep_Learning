# cnn_experiment

## Mục đích thí nghiệm
Kiểm tra xem việc tăng cường độ dropout có cải thiện khả năng tổng quát hóa (generalization) của CNN trên FashionMNIST so với cnn_baseline hay không.

## Giả thuyết
Tăng dropout từ 0.3 lên 0.5 sẽ giảm khoảng cách train/validation accuracy (ít overfitting hơn), có thể phải đánh đổi bằng tốc độ hội tụ chậm hơn một chút.

## Thay đổi so với baseline
dropout: 0.3 -> 0.5 (thay đổi một yếu tố duy nhất)

## Những gì được giữ nguyên
Kiến trúc (FashionCNN), số epoch, batch size, learning rate, optimizer, seed, và cách chia train/validation giống hệt cnn_baseline.

## Dự đoán kết quả
Khoảng cách giữa train_accuracy và val_accuracy nhỏ hơn so với cnn_baseline, với validation accuracy tương tự hoặc hơi khác.

## Kết quả quan sát được
- Epoch tốt nhất: 12 (validation loss thấp nhất)
- Validation loss tốt nhất: 0.2282
- Validation accuracy tốt nhất: 0.9262
- Train accuracy ở epoch cuối: 0.9480
- Validation accuracy ở epoch cuối: 0.9272
- Khoảng cách train/validation accuracy cuối cùng: 0.0208

## Diễn giải
Xem notebooks/02_training_and_comparison.ipynb để so sánh với các lần chạy khác.

## Kết luận
Độ chính xác trên tập test: 0.9204, tốt nhất trong cả ba lần chạy. Dropout cao hơn (0.5) đã giảm khoảng cách
train/validation accuracy (0.021 so với 0.040 của baseline) và cho một cải thiện nhỏ về validation/test accuracy,
ủng hộ giả thuyết. Được chọn làm mô hình cuối cùng cho phân tích lỗi (notebook 03).
