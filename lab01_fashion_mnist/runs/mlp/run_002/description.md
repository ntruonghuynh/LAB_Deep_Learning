# mlp_baseline

## Mục đích thí nghiệm
Thiết lập một mô hình cơ sở fully-connected đơn giản cho bài toán phân loại FashionMNIST, để so sánh với CNN baseline.

## Giả thuyết
Một MLP với 2 lớp ẩn sẽ đạt độ chính xác chấp nhận được nhưng kém hơn CNN, vì nó bỏ qua cấu trúc không gian 2D trong ảnh.

## Thay đổi so với baseline
không áp dụng - đây là lần chạy baseline đầu tiên

## Những gì được giữ nguyên
không áp dụng - đây là lần chạy baseline đầu tiên

## Dự đoán kết quả
Validation accuracy thấp hơn rõ rệt so với CNN baseline, với khoảng cách train/validation ở mức vừa phải tùy vào dropout.

## Kết quả quan sát được
- Epoch tốt nhất: 11 (validation loss thấp nhất)
- Validation loss tốt nhất: 0.3086
- Validation accuracy tốt nhất: 0.8870
- Train accuracy ở epoch cuối: 0.9103
- Validation accuracy ở epoch cuối: 0.8898
- Khoảng cách train/validation accuracy cuối cùng: 0.0204

## Diễn giải
Xem notebooks/02_training_and_comparison.ipynb để so sánh với các lần chạy khác.
