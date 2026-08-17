# mobilenet_v2_pretrained

## Mục đích
Thử nghiệm mô hình MobileNetV2 gọn nhẹ, được huấn luyện trước trên ImageNet, cho bài toán transfer learning trên FashionMNIST.

## Giả thuyết
MobileNetV2 được kỳ vọng có thời gian huấn luyện nhanh hơn các kiến trúc nặng hơn nhưng vẫn đạt độ chính xác validation tốt nhờ tận dụng các đặc trưng đã học từ ImageNet.

## Thay đổi so với baseline
Sử dụng mô hình `torchvision.models.mobilenet_v2` với trọng số pretrained, thay lớp classifier cuối bằng lớp đầu ra 10 lớp và đóng băng phần feature extractor.

## Các yếu tố được giữ cố định
Giữ nguyên cách chia dữ liệu FashionMNIST, seed, pipeline tiền xử lý ảnh 224×224 theo dạng 3 kênh và quy trình đánh giá như các thí nghiệm pretrained khác.

## Kết quả kỳ vọng
MobileNetV2 được kỳ vọng mang lại sự cân bằng tốt giữa tốc độ và độ chính xác, đặc biệt trong trường hợp huấn luyện trên CPU.
