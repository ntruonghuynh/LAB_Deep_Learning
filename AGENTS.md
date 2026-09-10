# Repository instructions

## Project scope

- Repository này gồm hai dự án độc lập: `lab01_fashion_mnist` (FashionMNIST classification từ đầu) và `lab02_transfer_learning` (transfer learning trên FashionMNIST).
- Luôn xác định lab đang được xử lý trước khi đọc, phân tích hoặc sửa file. Trong thư mục con, áp dụng thêm `AGENTS.md` của lab đó.
- Không dùng source, config, metric, run, checkpoint, notebook output hoặc artifact của lab này để chứng minh cho lab kia.
- Nếu yêu cầu liên quan cả hai lab, audit từng lab độc lập trước; chỉ so sánh sau khi đã xác định source và artifact hợp lệ của mỗi bên.

## Source-of-truth priority

Ưu tiên bằng chứng theo thứ tự:

1. Đề bài và yêu cầu hiện tại của người dùng.
2. Source code và script thực tế.
3. `config.yaml` được lưu trong từng run.
4. Artifact của run: CSV, JSON, checkpoint, PNG và TensorBoard event nếu có.
5. Notebook code và output đã lưu.
6. README, report, description và conclusion viết tay.

Nếu các nguồn mâu thuẫn, nêu rõ từng nguồn và nội dung mâu thuẫn. Không bỏ qua nguồn ưu tiên cao hơn hoặc chọn nguồn thuận tiện để hỗ trợ kết luận.

## Audit rules

Khi người dùng yêu cầu audit, review hoặc phân tích:

- Chỉ đọc; không sửa file hay tạo artifact trong repository.
- Không chạy training, không tải dataset/pretrained weights và không cài package.
- Chỉ chạy kiểm tra nhẹ, read-only và không làm đổi repository khi thật sự cần.
- Phân biệt rõ năm mức bằng chứng: code tồn tại; có call site; đã chạy; có artifact; đã được trình bày trong notebook/report.
- Dẫn bằng chứng bằng `file:dòng`, notebook cell hoặc đường dẫn artifact cụ thể. Nếu không tìm thấy, ghi rõ `Không tìm thấy`; không suy đoán.
- Không ước lượng metric bằng mắt khi có CSV/JSON. Đối chiếu plot với dữ liệu nguồn khi cần.
- Dùng các trạng thái `PASS`, `PARTIAL`, `FAIL`, `NOT VERIFIED` và giải thích tiêu chí áp dụng.
- Không gọi một lựa chọn thiết kế là bug nếu chưa có bằng chứng. Tách riêng: bug chắc chắn, rủi ro phương pháp luận, code smell và cải tiến tùy chọn.
- Kiểm tra artifact theo đúng lab và đúng run; không coi tên file hoặc mô tả viết tay là bằng chứng đủ về provenance.

## Change rules

Khi người dùng yêu cầu sửa:

- Chỉ sửa trong phạm vi được yêu cầu và đọc `git status --short` trước khi chỉnh sửa.
- Bảo toàn mọi thay đổi hiện có của người dùng; không xóa, revert hoặc ghi đè thay đổi ngoài phạm vi.
- Không thay đổi artifact của official run, không chạy lại experiment và không thay metric nếu chưa được yêu cầu rõ ràng.
- Giữ source, config, notebook và tài liệu của hai lab tách biệt.
- Sau khi sửa, chạy kiểm tra nhỏ nhất phù hợp; báo chính xác lệnh/kiểm tra đã chạy, kết quả và phần chưa kiểm tra.
- Không commit hoặc push trừ khi người dùng yêu cầu rõ ràng.

## Chart-analysis rules

Truy vết mỗi biểu đồ theo chuỗi:

```text
notebook cell
→ hàm visualization
→ CSV/JSON nguồn
→ training/evaluation code
→ config/run
```

- Gộp các bản trùng của cùng biểu đồ giữa notebook output, PNG, TensorBoard và biểu đồ vẽ lại; không đếm chúng như bằng chứng độc lập.
- Với mỗi biểu đồ, giải thích trục, đường/màu, dữ liệu nguồn, pattern, kết luận hợp lệ và giới hạn của kết luận.
- Không kết luận overfitting chỉ từ một epoch tăng đơn lẻ; xem xu hướng qua nhiều epoch và khoảng cách train/validation.
- Phân biệt best epoch với final epoch.
- Phân biệt maximum accuracy với accuracy tại epoch có minimum validation loss.
- Khi hình và CSV/JSON mâu thuẫn, ưu tiên dữ liệu máy đọc được và báo rõ sự không nhất quán.
