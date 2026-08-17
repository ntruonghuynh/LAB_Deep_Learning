# resnet18_pretrained

## Mục đích
Sử dụng mô hình ResNet18 được huấn luyện trước trên ImageNet làm baseline transfer learning cho bài toán phân loại FashionMNIST.

## Giả thuyết
ResNet18 pretrained với backbone được đóng băng được kỳ vọng có thể học nhanh phần classification head của FashionMNIST, ngay cả khi chỉ huấn luyện trong số epoch tương đối ít.

## Thay đổi so với baseline
Thay mô hình CNN tự xây dựng bằng `torchvision.models.resnet18` sử dụng trọng số pretrained, resize ảnh FashionMNIST lên 224×224, chuyển ảnh grayscale thành 3 kênh, đóng băng phần feature extractor và chỉ huấn luyện lớp classifier cuối.

## Các yếu tố được giữ cố định
Giữ nguyên bộ dữ liệu FashionMNIST, tỷ lệ validation, seed, hàm loss và quy trình đánh giá trên train/validation/test giống các thí nghiệm trước.

## Kết quả kỳ vọng
Validation accuracy được kỳ vọng cải thiện nhanh so với mô hình khởi tạo ngẫu nhiên, đồng thời quá trình huấn luyện tương đối ổn định vì chỉ có lớp classifier được cập nhật.

## Kết quả quan sát được
- Epoch tốt nhất: 5 (có validation loss thấp nhất)
- Validation loss tốt nhất: 0.3755
- Validation accuracy tốt nhất: 0.8638
- Train accuracy ở epoch cuối: 0.8587
- Validation accuracy ở epoch cuối: 0.8638
- Chênh lệch train/validation accuracy ở epoch cuối: -0.0051

## Nhận xét
Xem `notebooks/02_training_and_comparison.ipynb` để so sánh thí nghiệm này với các lần chạy khác.