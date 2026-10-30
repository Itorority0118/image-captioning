import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


# =========================================================
# 1. Special tokens
# =========================================================

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<start>"
EOS_TOKEN = "<end>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
]


# =========================================================
# 2. Tokenize caption
# =========================================================

def tokenize_caption(text):
    """
    Chuyển một caption thành danh sách token.
    Ví dụ: "A dog is running!"
    -> ["a", "dog", "is", "running", "!"]
    """
    if not isinstance(text, str):
        text = ""

    text = text.lower().strip()

    return re.findall(r"[a-z0-9]+|[^\w\s]", text)


# =========================================================
# 3. Load captions từ CSV Flickr30k
# =========================================================

def load_captions(csv_path):
    """
    Đọc CSV Flickr30k, chuẩn hóa tên cột và loại bỏ
    caption trống. Trả về DataFrame có image_name, comment.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy CSV: {csv_path}")

    df = pd.read_csv(
        csv_path,
        sep="|",
        skipinitialspace=True,
    )

    df.columns = df.columns.astype(str).str.strip()

    # Chuẩn hóa tên cột để tránh lỗi do khoảng trắng
    normalized_columns = {
        col: col.strip().lower()
        for col in df.columns
    }
    df = df.rename(columns=normalized_columns)

    required = {"image_name", "comment"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV cần có các cột {required}; "
            f"các cột hiện có: {df.columns.tolist()}"
        )

    df["image_name"] = (
        df["image_name"].fillna("").astype(str).str.strip()
    )
    df["comment"] = (
        df["comment"].fillna("").astype(str).str.strip()
    )

    df = df[
        (df["image_name"] != "")
        & (df["comment"] != "")
    ].copy()

    # Loại caption trùng của cùng một ảnh
    df = df.drop_duplicates(
        subset=["image_name", "comment"],
        keep="first",
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"Không có caption hợp lệ trong CSV: {csv_path}")

    return df[["image_name", "comment"]]


# =========================================================
# 4. Build vocabulary từ caption tập train
# =========================================================

def build_vocab(captions, min_freq=2):
    """
    Xây dựng vocabulary từ caption train.
    Các token đặc biệt luôn được đặt ở đầu vocabulary.
    """
    if min_freq < 1:
        raise ValueError("min_freq phải >= 1")

    counter = Counter()

    for caption in captions:
        counter.update(tokenize_caption(caption))

    vocab = {
        token: idx
        for idx, token in enumerate(SPECIAL_TOKENS)
    }

    for token, frequency in counter.most_common():
        if frequency >= min_freq and token not in vocab:
            vocab[token] = len(vocab)

    return vocab


# =========================================================
# 5. Encode caption thành các ID
# =========================================================

def encode_caption(caption, vocab, max_len=40):
    """
    Chuyển caption thành dãy ID:
    <start> + tokens + <end> + <pad>

    Kết quả có độ dài chính xác max_len.
    """
    if max_len < 2:
        raise ValueError("max_len phải >= 2 để chứa <start> và <end>")

    for token in SPECIAL_TOKENS:
        if token not in vocab:
            raise ValueError(
                f"Vocabulary thiếu token bắt buộc: {token}"
            )

    tokens = tokenize_caption(caption)

    start_id = vocab[BOS_TOKEN]
    end_id = vocab[EOS_TOKEN]
    pad_id = vocab[PAD_TOKEN]
    unk_id = vocab[UNK_TOKEN]

    # Chừa vị trí cho <start> và <end>
    tokens = tokens[:max_len - 2]

    token_ids = [
        vocab.get(token, unk_id)
        for token in tokens
    ]

    ids = [start_id] + token_ids + [end_id]

    if len(ids) < max_len:
        ids.extend([pad_id] * (max_len - len(ids)))

    return ids


# =========================================================
# 6. Decode ID thành caption
# =========================================================

def decode_caption(ids, vocab):
    """
    Chuyển ID về caption, bỏ qua token đặc biệt.
    Dừng khi gặp <end>.
    """
    idx_to_token = {
        int(idx): token
        for token, idx in vocab.items()
    }

    tokens = []

    for value in ids:
        # Hỗ trợ cả số nguyên Python và Tensor scalar
        if hasattr(value, "item"):
            value = value.item()

        token = idx_to_token.get(int(value), UNK_TOKEN)

        if token == EOS_TOKEN:
            break

        if token in SPECIAL_TOKENS:
            continue

        tokens.append(token)

    # Ghép dấu câu sát với từ phía trước
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)

    return text.strip()


# =========================================================
# 7. Save vocabulary
# =========================================================

def save_vocab(vocab, output_path):
    """Lưu vocabulary thành JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


# =========================================================
# 8. Load vocabulary
# =========================================================

def load_vocab(vocab_path):
    """Đọc vocabulary từ JSON."""
    vocab_path = Path(vocab_path)

    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy vocabulary: {vocab_path}"
        )

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab = {
        token: int(idx)
        for token, idx in vocab.items()
    }

    for token in SPECIAL_TOKENS:
        if token not in vocab:
            raise ValueError(
                f"Vocabulary thiếu token bắt buộc: {token}"
            )

    return vocab