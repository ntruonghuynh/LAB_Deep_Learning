# Thư mục Data

Thư mục này chứa dữ liệu được lưu đệm cục bộ. Nó không phải là một phần của mã nguồn.

- `FashionMNIST/` — các file dữ liệu gốc được tải tự động bởi `torchvision.datasets.FashionMNIST`
  trong lần chạy đầu tiên (thông qua `src/dataset.py`). Thư mục này bị gitignore vì nó có thể
  được tạo lại bằng cách chạy lại mã nguồn; không commit nó.
- `splits/` — các split chỉ số (index) train/validation cố định (JSON), được tạo một lần với một
  seed cố định để mọi thí nghiệm (MLP, CNN, CNN experiment) huấn luyện/kiểm định trên đúng cùng
  một dữ liệu. Các file này ĐƯỢC commit vì chúng định nghĩa split có thể tái lập được dùng chung
  cho mọi lần chạy chính thức.

## Vì sao cần một file split cố định?

FashionMNIST đi kèm với một tập huấn luyện chính thức (60.000 ảnh) và một tập kiểm tra chính
thức (10.000 ảnh). Tập kiểm tra chỉ được phép dùng để đánh giá cuối cùng (xem `README.md` ở
gốc dự án). Chúng ta chia tập huấn luyện chính thức thành một tập con huấn luyện và một tập
con validation một lần duy nhất, lưu lại các chỉ số kết quả vào `data/splits/split_seed42.json`,
và tái sử dụng các chỉ số đó cho mọi lần chạy. Điều này tránh việc vô tình so sánh các mô hình
được huấn luyện/kiểm định trên dữ liệu khác nhau.
