# resnet18_finetune

## Mục đích
Fine-tune khối convolution cuối cùng của mô hình ResNet18 đã được huấn luyện trước trên ImageNet cho bài toán phân loại FashionMNIST.

## Giả thuyết
Việc mở khóa khối ResNet cuối cùng được kỳ vọng giúp mô hình thích nghi tốt hơn với FashionMNIST, đồng thời vẫn tận dụng được các đặc trưng thị giác đã học từ ImageNet.

## Thay đổi so với baseline
Bắt đầu từ cấu hình ResNet18 pretrained với backbone được đóng băng, sau đó mở khóa `layer4` và sử dụng learning rate nhỏ hơn để fine-tune khối đặc trưng cuối cùng cùng với lớp classifier.

## Các yếu tố được giữ cố định
Giữ nguyên bộ dữ liệu, kích thước ảnh, ImageNet normalization, batch size, seed và quy trình đánh giá giống với ResNet18 pretrained baseline.

## Kết quả kỳ vọng
Fine-tuning có thể giúp cải thiện validation accuracy so với baseline chỉ huấn luyện classifier, nhưng cũng có thể dẫn đến overfitting nhanh hơn nếu huấn luyện quá nhiều epoch.

## Kết quả quan sát được
- Epoch tốt nhất: 4 (có validation loss thấp nhất)
- Validation loss tốt nhất: 0.1929
- Validation accuracy tốt nhất: 0.9320
- Train accuracy ở epoch cuối: 0.9715
- Validation accuracy ở epoch cuối: 0.9275
- Chênh lệch train/validation accuracy ở epoch cuối: 0.0440

## Nhận xét
Xem `notebooks/02_training_and_comparison.ipynb` để so sánh thí nghiệm này với các lần chạy khác.