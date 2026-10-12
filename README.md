# Image Captioning: EfficientNet-B0 + GPT-2

Dự án tạo câu mô tả cho ảnh.
Dataset: Flickr30k. Encoder: EfficientNet-B0. Decoder: GPT-2.

---

## A. Chạy trên máy cá nhân (Windows)

### Bước 1: Cài các phần mềm cần thiết (làm một lần)

1. **Python** (bản 3.10, 3.11 hoặc 3.12): tải tại python.org. Khi cài, **nhớ tick ô "Add Python to PATH"**.
2. **VS Code** (nơi viết code): tải tại code.visualstudio.com.
3. Chọn **một** trong hai công cụ để tải code và nộp code:
   - **Cách 1, GitHub Desktop** (có giao diện, dễ dùng): tải tại desktop.github.com, rồi đăng nhập tài khoản GitHub.
   - **Cách 2, Git** (gõ lệnh): tải tại git-scm.com, cài với các lựa chọn mặc định. Cài xong gõ `git --version` trong terminal, thấy hiện số phiên bản là được.

### Bước 2: Tải code về máy

**Cách 1: dùng GitHub Desktop**

1. Mở GitHub Desktop → **File → Clone repository**.
2. Chọn tab **URL**, dán: `https://github.com/Itorority0118/image-captioning.git`
3. Chọn thư mục lưu trên máy → bấm **Clone**.
4. Bấm **Open in Visual Studio Code**.

**Cách 2: dùng Git**

Mở terminal (PowerShell hoặc Git Bash), đi tới thư mục muốn lưu dự án, rồi gõ:

    git clone https://github.com/Itorority0118/image-captioning.git
    cd image-captioning
    code .

Lệnh `code .` mở thư mục này bằng VS Code.

### Bước 3: Mở terminal

Trong VS Code, bấm **Terminal → New Terminal**. Cửa sổ lệnh hiện ở dưới. Các lệnh bên dưới đều gõ vào đây, mỗi lệnh gõ xong nhấn Enter.

### Bước 4: Tạo môi trường ảo

    python -m venv .venv
    .venv\Scripts\activate

Thành công khi đầu dòng lệnh có chữ `(.venv)`.

Nếu báo lỗi "running scripts is disabled": mở PowerShell, gõ
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, chọn Y, rồi quay lại làm lại bước này.

### Bước 5: Cài thư viện

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt

### Bước 6: Chọn interpreter cho VS Code

Nhấn `Ctrl+Shift+P`, gõ `Python: Select Interpreter`, chọn mục có chữ `.venv`.

### Bước 7: Kiểm tra

    python -c "from src.paths import ENV; print(ENV)"

Thành công khi màn hình in ra `local`.

---

## B. Chạy trên Google Colab

1. Vào colab.research.google.com → **New notebook**.
2. Bật GPU: **Runtime → Change runtime type → T4 GPU → Save**.
3. Dán đoạn sau vào ô code đầu tiên rồi bấm nút chạy:

```
!git clone https://github.com/Itorority0118/image-captioning.git
%cd image-captioning
!pip install -q -r requirements.txt
!python -c "from src.paths import ENV; import torch; print(ENV, torch.cuda.is_available())"
```

Thành công khi in ra `colab True`.
Khi code trên GitHub có cập nhật mới, mở lại notebook và chạy `!git pull` (nếu phiên cũ còn) hoặc chạy lại ô clone ở trên (nếu mở phiên mới).

---

## C. Dữ liệu

Dataset Flickr30k **không nằm trong repo** (quá nặng).

- Tải dataset tại: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset (cần đăng nhập Kaggle)
- **Máy cá nhân:** giải nén vào thư mục `data/` ở gốc dự án.
- **Colab:** Colab mất dữ liệu mỗi khi đóng phiên, nên cách dễ nhất là upload file zip lên Google Drive một lần, rồi mỗi phiên chạy ô này (đổi tên file zip cho đúng):

```
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/data
!unzip -q /content/drive/MyDrive/flickr30k.zip -d /content/data
```

---

## C2. Cấu trúc thư mục

```
image-captioning/
├── src/                    Code chính của dự án
│   ├── paths.py            Tự nhận biết đang chạy local hay Colab, chọn thư mục phù hợp
│   ├── train.py            Chạy huấn luyện mô hình
│   ├── data/               Xử lý dữ liệu (ảnh, caption, Dataset)
│   ├── models/             Định nghĩa mô hình (EfficientNet-B0 + GPT-2)
│   ├── decoding/           Thuật toán sinh câu (Greedy Search, Beam Search)
│   └── eval/               Tính điểm BLEU, ROUGE, METEOR, CIDEr, SPICE
├── app/                    Web demo (Gradio)
├── notebooks/              Notebook chạy trên Colab
├── configs/
│   └── base.yaml           Cấu hình huấn luyện (learning rate, batch size, ...)
├── tests/                  Code kiểm tra tự động
├── requirements.txt        Danh sách thư viện cần cài
├── .gitignore              Danh sách file/thư mục không đưa lên GitHub
└── README.md               File hướng dẫn này
```

Các thư mục dưới đây **không nằm trên GitHub** (vì quá nặng):

| Thư mục | Nội dung | Cách có được |
|---|---|---|
| `data/` | Ảnh và caption gốc của Flickr30k | Tự tải về và bỏ vào (xem mục C) |
| `outputs/features/` | Đặc trưng ảnh đã trích xuất (file `.npy`) | Tự tạo khi chạy code |
| `outputs/checkpoints/` | Mô hình đã huấn luyện (file `.pt`) | Tự tạo khi chạy code |

Trên Colab, hai thư mục kết quả nằm ở `/content/work/features` và `/content/work/checkpoints`.

Lưu ý: `data/` ở gốc dự án (chứa dataset) khác với `src/data/` (chứa code xử lý dữ liệu).

### Ai phụ trách thư mục nào

| Thành viên | Thư mục / file phụ trách |
|---|---|
| Thành viên 1 (ảnh + feature) | `src/data/image_features.py`, `src/data/split.py` |
| Thành viên 2 (văn bản + Dataset) | `src/data/text_preprocess.py`, `src/data/dataset.py` |
| Thành viên 3 (mô hình + huấn luyện) | `src/models/`, `src/decoding/`, `src/train.py`, `configs/` |
| Thành viên 4 (đánh giá + web) | `src/eval/`, `app/`, `notebooks/` |

Quy tắc: **chỉ sửa file thuộc phần mình phụ trách**. Cần sửa file của người khác thì nhắn họ.

---

## D. Quy tắc làm việc nhóm

Quy tắc chung (dùng cách nào cũng phải theo):

- Không commit thẳng vào nhánh `main`.
- Viết xong thì mở Pull Request, chờ duyệt rồi mới merge.
- Không đưa mật khẩu, token hay API key vào code (repo này công khai).

### Cách 1: dùng GitHub Desktop

1. Tạo nhánh: **Current branch → New branch**, đặt tên, ví dụ `feat/image-pipeline`.
2. Viết code trong VS Code.
3. Quay lại GitHub Desktop: ghi mô tả ở ô Summary → **Commit** → **Publish branch** → **Create Pull Request**.
4. Chờ duyệt và merge.
5. Lấy code mới nhất: chọn nhánh `main` → bấm **Fetch origin** → **Pull origin**.

### Cách 2: dùng Git (gõ lệnh)

1. Lấy code mới nhất trước khi bắt đầu:

       git checkout main
       git pull

2. Tạo nhánh mới và chuyển sang nhánh đó (đổi tên nhánh cho phù hợp):

       git checkout -b feat/image-pipeline

3. Viết code trong VS Code. Xong thì lưu lại và đẩy lên GitHub:

       git status
       git add .
       git commit -m "mo ta ngan gon viec da lam"
       git push -u origin feat/image-pipeline

4. Mở trang repo trên github.com, bấm nút **Compare & pull request** → **Create pull request**.
5. Chờ duyệt và merge. Sau đó quay về nhánh chính:

       git checkout main
       git pull