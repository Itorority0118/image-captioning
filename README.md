# Image Captioning: so sánh 4 mô hình sinh chú thích ảnh

Bài toán: tự động sinh chú thích tiếng Anh cho ảnh.
Dataset: Flickr30k.

4 mô hình được so sánh:

| Tên mô hình | Encoder (đọc ảnh) | Decoder (viết câu) |
|---|---|---|
| `mobilenetv3_lstm` | MobileNetV3-Small | LSTM + Attention |
| `efficientnet_b0_gpt2` | EfficientNet-B0 | GPT-2 |
| `deit_tiny_gpt2` | DeiT-Tiny | GPT-2 |
| `tinyclip_gpt2` | TinyCLIP (ViT-40M/32) | GPT-2 |

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

- Link dataset: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset (tải tay thì cần đăng nhập Kaggle)
- **Máy cá nhân:** không cần cả bộ, chỉ cần bộ ảnh nhỏ để thử code (xem phần "Bộ ảnh nhỏ" bên dưới).
- **Colab:** không cần tải tay hay chép file qua Drive. Ô lệnh dưới đây tự tải dataset từ Kaggle về ổ đĩa Colab (nhanh hơn nhiều so với đọc ảnh từ Drive):

```
!pip install -q kagglehub
import kagglehub
path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")
```

Biến `path` giữ vị trí dataset. Ảnh nằm ở `{path}/flickr30k_images/flickr30k_images`, caption ở `{path}/flickr30k_images/results.csv`. (Dataset trên Kaggle bị đóng gói hai bản ảnh lồng nhau, nên đường dẫn có hai tầng `flickr30k_images`.)

Lưu ý: `{path}` chỉ có giá trị khi dòng `path = ...` đã chạy trong cùng phiên, nên luôn chạy cả ô từ trên xuống dưới.

### Lưu feature và checkpoint lên Google Drive (Colab, không bắt buộc)

Feature và mô hình đã train **mất khi đóng phiên Colab**. Feature trích lại chỉ mất vài phút, nhưng checkpoint đã train thì nên cất sang Drive **của chính mình**. Đừng chép thẳng thư mục feature (hàng chục nghìn file nhỏ, Drive xử lý rất chậm), hãy nén thành một file zip. Thay `ENCODER` bằng tên encoder của bạn:

```
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/captioning
!cd /content/work/features && zip -r -0 -q /content/features_ENCODER.zip ENCODER
!cp /content/features_ENCODER.zip /content/drive/MyDrive/captioning/
!cp -r /content/work/checkpoints /content/drive/MyDrive/captioning/
```

Phiên mới muốn dùng lại:

```
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/work/features
!cp /content/drive/MyDrive/captioning/features_ENCODER.zip /content/
!unzip -q /content/features_ENCODER.zip -d /content/work/features
!cp -r /content/drive/MyDrive/captioning/checkpoints /content/work/
```

### Bộ ảnh nhỏ để test code trên máy cá nhân

Không cần giữ cả dataset trên máy. Chỉ cần vài trăm ảnh để chạy thử code:

1. Tải dataset đầy đủ về máy một lần (link ở trên) và giải nén ra một thư mục tạm.
2. Mở `src/data/make_sample.py`, sửa 3 dòng đầu (`CSV_PATH`, `IMAGE_DIR`, `NUM_IMAGES`) cho đúng đường dẫn trên máy.
3. Chạy lệnh:

```
python src/data/make_sample.py
```

Script lấy ngẫu nhiên 300 ảnh (seed 42, nên cả nhóm ra cùng một bộ) và chép vào thư mục `data/`.

---

## C2. Cấu trúc thư mục

```
image-captioning/
├── src/                      Code chính của dự án
│   ├── paths.py              Tự nhận biết đang chạy local hay Colab, chọn thư mục phù hợp
│   ├── train.py              Chạy huấn luyện (chọn encoder và decoder)
│   ├── data/                 Xử lý dữ liệu (ảnh, caption, Dataset)
│   │   ├── make_sample.py    Tạo bộ ảnh nhỏ (vài trăm ảnh) để test code trên máy cá nhân
│   │   ├── split.py          Chia train/val/test theo tên ảnh
│   │   ├── explore_images.py Khám phá ảnh: kích thước, ảnh lỗi, so khớp với caption
│   │   └── image_features.py Trích feature ảnh bằng 4 encoder
│   ├── models/
│   │   ├── encoders.py       4 encoder (MobileNetV3, EfficientNet-B0, DeiT-Tiny, TinyCLIP)
│   │   ├── decoder_lstm.py   Decoder LSTM + Attention
│   │   └── decoder_gpt2.py   Decoder GPT-2
│   ├── decoding/
│   │   ├── greedy.py         Greedy Search
│   │   └── beam.py           Beam Search
│   └── eval/                 Tính điểm BLEU, ROUGE, METEOR, CIDEr, SPICE
├── app/                      Web demo (Gradio)
├── notebooks/                Notebook chạy trên Colab
├── configs/
│   ├── base.yaml             Cấu hình huấn luyện (learning rate, batch size, ...)
│   ├── splits.json           Danh sách ảnh train/val/test (cả nhóm dùng chung)
│   └── splits_sample.json    Bản chia cho bộ ảnh nhỏ (dùng để test trên máy)
├── tests/                    Code kiểm tra tự động
├── requirements.txt          Danh sách thư viện cần cài
├── .gitignore                Danh sách file/thư mục không đưa lên GitHub
└── README.md                 File hướng dẫn này
```

Các thư mục dưới đây **không nằm trên GitHub** (vì quá nặng):

| Thư mục | Nội dung | Cách có được |
|---|---|---|
| `data/` | Ảnh và caption gốc của Flickr30k | Tự tải về và bỏ vào (xem mục C) |
| `outputs/features/<encoder>/` | Feature ảnh (file `.npy`), mỗi encoder một thư mục | Tự tạo khi chạy code |
| `outputs/checkpoints/<encoder>_<decoder>/` | Mô hình đã train, mỗi mô hình một thư mục | Tự tạo khi chạy code |

Trên Colab, hai thư mục kết quả nằm ở `/content/work/features` và `/content/work/checkpoints`.

Lưu ý: `data/` ở gốc dự án (chứa dataset) khác với `src/data/` (chứa code xử lý dữ liệu).

### Tên chuẩn dùng trong toàn dự án

Viết đúng y như bảng này (trong code, tên thư mục, dòng lệnh):

| Encoder | Decoder |
|---|---|
| `mobilenetv3` | `lstm` |
| `efficientnet_b0` | `gpt2` |
| `deit_tiny` | |
| `tinyclip` | |

Ví dụ: thư mục feature `outputs/features/deit_tiny/`, thư mục checkpoint `outputs/checkpoints/deit_tiny_gpt2/`.

### Hình dạng feature của từng encoder (mỗi ảnh)

| Encoder | Shape | Dung lượng 31.783 ảnh (float16) |
|---|---|---|
| `mobilenetv3` | 49 x 576 | khoảng 1,8 GB |
| `efficientnet_b0` | 49 x 1280 | khoảng 4,0 GB |
| `deit_tiny` | 197 x 192 | khoảng 2,4 GB |
| `tinyclip` | 50 x 512 | khoảng 1,6 GB |

Với `tinyclip`, 50 vị trí gồm 1 vector chung và 49 mảnh ảnh (lấy `last_hidden_state`).
Decoder **không được ghi cứng số vị trí** (49, 197, 50): phải đọc từ shape của feature.

### Ai phụ trách thư mục nào

| Thành viên | File phụ trách | Mô hình tự huấn luyện |
|---|---|---|
| Thành viên 1 | `src/data/split.py`, `src/data/image_features.py`, `src/models/encoders.py`, `app/` | `tinyclip_gpt2` |
| Thành viên 2 | `src/data/text_preprocess.py`, `src/data/dataset.py`, `src/models/decoder_lstm.py`, `tests/` | `mobilenetv3_lstm` |
| Thành viên 3 | `src/models/decoder_gpt2.py`, `src/decoding/greedy.py`, `src/train.py`, `configs/` | `efficientnet_b0_gpt2` |
| Thành viên 4 | `src/eval/`, `src/decoding/beam.py`, `notebooks/` | `deit_tiny_gpt2` |

Quy tắc: **chỉ sửa file thuộc phần mình phụ trách**. Cần sửa file của người khác thì nhắn người đó.

---

## C3. Chạy tiền xử lý ảnh

`configs/splits.json` đã có sẵn trong repo (31.783 ảnh: 30.783 train, 1.000 val, 1.000 test, chia theo tên ảnh, seed 42), không cần chạy lại.

`--encoder` chọn một trong: `mobilenetv3`, `efficientnet_b0`, `deit_tiny`, `tinyclip`. Ảnh nào đã có file feature thì được bỏ qua, nên Colab bị ngắt thì chạy lại đúng lệnh cũ là làm tiếp.

**Thử nhanh trên máy cá nhân** (bộ ảnh nhỏ, chạy trong terminal ở thư mục gốc dự án, nhớ có `-m` và không có `.py`):

```
python -m src.data.image_features --encoder deit_tiny --splits configs/splits_sample.json
```

**Chạy toàn bộ trên Colab** (bật GPU trước, chạy cả ô từ trên xuống dưới, đổi `ENCODER` cho đúng):

```
!pip install -q kagglehub timm transformers
import kagglehub
path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")
%cd /content
!rm -rf image-captioning
!git clone https://github.com/Itorority0118/image-captioning.git
%cd image-captioning
!python -m src.data.image_features --encoder ENCODER --image_dir "{path}/flickr30k_images/flickr30k_images"
```

Nếu code chưa được merge vào `main`, thêm `-b tên-nhánh` sau chữ `clone`. Kết quả đúng: dòng đầu có `Thiết bị: cuda` và `Số ảnh trong danh sách: 31783`, dòng cuối có `OK`. Feature nằm ở `/content/work/features/ENCODER/`.

---

## D. Quy tắc làm việc nhóm

Quy tắc chung (dùng cách nào cũng phải theo):

- Mỗi người làm trên **một nhánh riêng của mình** (đặt tên theo mã số và tên, ví dụ `23130388_MaNguyenVu`). Mọi việc của mình đều commit vào nhánh đó, không tạo nhánh theo từng việc.
- Không commit thẳng vào nhánh `main`.
- Viết xong thì mở Pull Request, chờ duyệt rồi mới merge.
- Không đưa mật khẩu, token hay API key vào code (repo này công khai).

### Cách 1: dùng GitHub Desktop

1. Tạo nhánh của bạn một lần duy nhất: **Current branch → New branch**, đặt tên theo mã số và tên của bạn. Các lần sau làm tiếp trên nhánh đó.
2. Viết code trong VS Code.
3. Quay lại GitHub Desktop: ghi mô tả ở ô Summary → **Commit** → **Publish branch** (lần đầu) hoặc **Push origin** (các lần sau) → **Create Pull Request** khi cần gộp vào `main`.
4. Chờ duyệt và merge.
5. Lấy code mới nhất của `main`: chọn nhánh `main` → bấm **Fetch origin** → **Pull origin**, rồi quay lại nhánh của bạn.

### Cách 2: dùng Git (gõ lệnh)

1. Lấy code mới nhất trước khi bắt đầu:

       git checkout main
       git pull

2. Tạo nhánh của bạn một lần duy nhất (đổi thành mã số và tên của bạn). Các lần sau chỉ cần `git checkout <tên nhánh>`:

       git checkout -b 23130388_MaNguyenVu

3. Viết code trong VS Code. Xong thì lưu lại và đẩy lên GitHub:

       git status
       git add .
       git commit -m "mo ta ngan gon viec da lam"
       git push -u origin 23130388_MaNguyenVu

4. Mở trang repo trên github.com, bấm nút **Compare & pull request** → **Create pull request**.
5. Chờ duyệt và merge. Sau đó quay về nhánh chính để lấy code mới:

       git checkout main
       git pull