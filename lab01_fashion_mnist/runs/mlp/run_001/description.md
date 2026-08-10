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
- Validation loss tốt nhất: 0.3018
- Validation accuracy tốt nhất: 0.8902
- Train accuracy ở epoch cuối: 0.9128
- Validation accuracy ở epoch cuối: 0.8903
- Khoảng cách train/validation accuracy cuối cùng: 0.0224

## Diễn giải
Xem notebooks/02_training_and_comparison.ipynb để so sánh với các lần chạy khác.

## Kết luận
Độ chính xác trên tập test: 0.8847. Xác nhận MLP là một baseline hợp lý (~88-89%) nhưng bị vượt qua
bởi cả hai cấu hình CNN, phù hợp với giả thuyết rằng một MLP dựa trên flatten không thể khai thác
cấu trúc không gian 2D hiệu quả như convolution.
