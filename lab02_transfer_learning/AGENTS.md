# Lab 2 instructions

## Scope

- Chỉ áp dụng cho `lab02_transfer_learning` và bổ sung cho `../AGENTS.md`.
- Mục tiêu là transfer learning trên FashionMNIST bằng các backbone pretrained.
- Không dùng MLP/CNN từ đầu, config, summary row hoặc artifact Lab 1 làm kết quả Lab 2.

## Lab 2 audit focus

- Kiểm tra các trường hợp MobileNetV2 pretrained, ResNet18 frozen và ResNet18 fine-tune bằng source, run config và artifact thực tế.
- Xác minh preprocessing: resize `224×224`, grayscale thành 3 channel và ImageNet normalization cho pretrained backbone; validation/test không nhận augmentation ngoài ý muốn.
- Xác minh enum pretrained weights và việc official training thực sự yêu cầu/nạp weights; khi evaluate, kiểm tra model được dựng tương thích mà không tải weights lại không cần thiết.
- Truy vết freeze/unfreeze qua `requires_grad`; nêu chính xác module nào trainable trong từng config và phạm vi fine-tune thực tế.
- Kiểm tra BatchNorm khi backbone frozen: `requires_grad=False` không tự ngăn running statistics cập nhật nếu toàn model ở train mode.
- Xác minh optimizer chỉ nhận trainable parameters; kiểm tra learning rate, augmentation và checkpoint criterion theo từng run.
- Phân biệt best-loss epoch, maximum-accuracy epoch và final epoch; không gọi accuracy tại best-loss epoch là maximum accuracy.
- Kiểm tra model selection chỉ dùng validation và test chỉ được đánh giá sau khi chọn model. Nếu nhiều candidate đã có test result, ghi nhận nguy cơ selection bias.
- Truy vết training curves và TensorBoard tới history/config; kiểm tra model comparison, confusion matrix, per-class metrics, predicted/misclassified displays và error analysis.

## Official-run protection

- Chỉ coi một run hoàn chỉnh là hợp lệ khi có config, history, final metrics và checkpoint cần thiết; kiểm tra nội dung artifact thay vì chỉ nhìn tên thư mục.
- Bỏ qua run dở dang hoặc báo riêng là incomplete; không dùng nó trong so sánh kết quả chính thức.
- Không dùng các dòng MLP/CNN bị lẫn trong Lab 2 experiment summary. Đối chiếu summary với các thư mục run Lab 2 trước khi kết luận.
- Không sửa, tái tạo hoặc ghi đè official-run artifact nếu chưa được người dùng yêu cầu rõ ràng.
- Không hard-code metric có thể thay đổi; luôn đọc lại CSV/JSON nguồn của run đang xét.
