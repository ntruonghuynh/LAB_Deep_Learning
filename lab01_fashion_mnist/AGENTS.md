# Lab 1 instructions

## Scope

- Chỉ áp dụng cho `lab01_fashion_mnist` và bổ sung cho `../AGENTS.md`.
- Mục tiêu là PyTorch FashionMNIST classification với mô hình xây dựng từ đầu.
- Không dùng ResNet18, MobileNetV2, pretrained weights hoặc bất kỳ artifact nào của Lab 2 để chứng minh kết quả Lab 1.

## Assignment requirements

Khi audit, đối chiếu đầy đủ các yêu cầu: load FashionMNIST; apply transforms; build neural network; training loop có forward, loss, backward và optimizer step; evaluate accuracy; save và load model; experiment với architecture hoặc hyperparameter; visualize loss; hiển thị predicted versus actual; bàn giao Python code, brief report, loss graphs và image displays.

## Lab 1 audit focus

- Kiểm tra cả MLP và CNN: input, shape qua từng layer, output 10 lớp, trainable parameters và model factory.
- Xác nhận model trả logits; kiểm tra không đặt Softmax sai trước `CrossEntropyLoss`.
- Truy vết official train/test, cách chia train/validation, overlap, loader shuffle và transform riêng của train/validation/test.
- Kiểm tra nguy cơ stale split cache: cache có phụ thuộc và xác minh đầy đủ `seed`, `validation_ratio` và dataset size hay không.
- Kiểm tra `model.train()`, `model.eval()`, `no_grad`/`inference_mode`, dropout và gradient trong validation.
- Kiểm tra loss/accuracy được cộng và chia theo số mẫu, kể cả batch cuối nhỏ hơn.
- Xác định checkpoint criterion; phân biệt best-loss epoch, maximum-accuracy epoch và final epoch. Xác minh test dùng best checkpoint hay state cuối.
- Với các run, kiểm tra yếu tố thay đổi và các yếu tố được giữ cố định; không quy nguyên nhân cho một biến nếu nhiều biến cùng thay đổi.
- Ghi nhận nguy cơ selection bias nếu test result của nhiều model đã được xem trước khi chọn model cuối; không đánh đồng việc này với training leakage.
- Truy vết loss/accuracy curves, confusion matrix, per-class metrics, misclassified samples và predicted-versus-actual display tới call site và dữ liệu nguồn.
- Xác định augmentation có được bật trong official run hay chỉ xuất hiện trong code/notebook minh họa.
- Không hard-code metric trong file hướng dẫn hoặc kết luận audit; luôn đọc lại artifact của official run đang xét.
