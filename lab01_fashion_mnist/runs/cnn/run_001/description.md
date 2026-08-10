# cnn_baseline

## Mục đích thí nghiệm
Thiết lập một CNN baseline đơn giản cho bài toán phân loại FashionMNIST, để so sánh với MLP baseline và dùng làm tham chiếu cho CNN experiment.

## Giả thuyết
Một CNN nhỏ sẽ vượt trội hơn MLP baseline bằng cách khai thác cấu trúc không gian cục bộ (cạnh, kết cấu) thông qua convolution và pooling.

## Thay đổi so với baseline
không áp dụng - đây là lần chạy tham chiếu/baseline của CNN

## Những gì được giữ nguyên
không áp dụng - đây là lần chạy tham chiếu/baseline của CNN

## Dự đoán kết quả
Validation accuracy cao hơn MLP baseline với cùng ngân sách huấn luyện (số epoch, batch size, learning rate, optimizer, cách chia dữ liệu).

## Kết quả quan sát được
- Epoch tốt nhất: 7 (validation loss thấp nhất)
- Validation loss tốt nhất: 0.2270
- Validation accuracy tốt nhất: 0.9195
- Train accuracy ở epoch cuối: 0.9661
- Validation accuracy ở epoch cuối: 0.9257
- Khoảng cách train/validation accuracy cuối cùng: 0.0405

## Diễn giải
Xem notebooks/02_training_and_comparison.ipynb để so sánh với các lần chạy khác.

## Kết luận
Độ chính xác trên tập test: 0.9161. Vượt trội rõ rệt so với MLP baseline. Train accuracy tiếp tục tăng lên tới 0.966
trong khi val accuracy chững lại quanh 0.92-0.93 sau epoch 7, cho thấy overfitting nhẹ - đây là động lực cho
thí nghiệm dropout (cnn_experiment).
