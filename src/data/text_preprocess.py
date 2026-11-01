import re
import html
from pathlib import Path
from functools import lru_cache

import pandas as pd
from transformers import GPT2Tokenizer


TOKENIZER_NAME = "gpt2"
MAX_LEN = 40

BOS_TOKEN = "<|endoftext|>"
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|endoftext|>"

GPT2_VOCAB_SIZE = 50257
SPECIAL_TOKEN_ID = 50256


@lru_cache(maxsize=1)
def load_gpt2_tokenizer():
    """
    Load GPT-2 BPE tokenizer.

    BOS, EOS and PAD use the same token:
    <|endoftext|>, ID 50256.

    No new token is added to GPT-2 vocabulary.
    """

    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_NAME)

    tokenizer.bos_token = BOS_TOKEN
    tokenizer.eos_token = EOS_TOKEN
    tokenizer.pad_token = PAD_TOKEN

    if tokenizer.vocab_size != GPT2_VOCAB_SIZE:
        raise ValueError(
            f"GPT-2 vocabulary phải có {GPT2_VOCAB_SIZE} tokens, "
            f"nhận được {tokenizer.vocab_size}"
        )

    if not (
        tokenizer.bos_token_id
        == tokenizer.eos_token_id
        == tokenizer.pad_token_id
        == SPECIAL_TOKEN_ID
    ):
        raise ValueError("BOS, EOS, PAD phải cùng có ID 50256.")

    return tokenizer



def clean_caption(text):
    """
    Light cleaning for Flickr30k captions.

    - Decode HTML entities.
    - Remove invalid control characters.
    - Normalize whitespace.
    - Preserve English capitalization and punctuation.
    - Do not perform word-level tokenization.
    """

    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

    text = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
        "",
        text,
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_captions(csv_path):
    """
    Read Flickr30k results.csv.

    Return DataFrame with:
        image_name
        comment

    Empty captions and exact duplicate image-caption pairs
    are removed.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy CSV: {csv_path}"
        )

    df = pd.read_csv(
        csv_path,
        sep="|",
        skipinitialspace=True,
    )

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
    )

    required = {"image_name", "comment"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV cần có các cột {required}; "
            f"các cột hiện có: {df.columns.tolist()}"
        )

    df = df[["image_name", "comment"]].copy()

    df["image_name"] = (
        df["image_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["comment"] = (
        df["comment"]
        .fillna("")
        .apply(clean_caption)
    )

    df = df[
        (df["image_name"] != "")
        & (df["comment"] != "")
    ].copy()

    df = df.drop_duplicates(
        subset=["image_name", "comment"],
        keep="first",
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"Không có caption hợp lệ trong CSV: {csv_path}"
        )

    return df


def encode_caption(
    caption,
    tokenizer=None,
    max_len=MAX_LEN,
):
    """
    Encode one caption using GPT-2 BPE.

    Sequence:
        BOS + BPE tokens + EOS + PAD

    Return:
        input_ids:      (max_len,)
        attention_mask: (max_len,)
        length:         number of non-padding tokens

    Attention mask is created BEFORE padding because
    EOS and PAD share token ID 50256.
    """

    if max_len < 2:
        raise ValueError("max_len phải >= 2.")

    if tokenizer is None:
        tokenizer = load_gpt2_tokenizer()

    caption = clean_caption(caption)

    token_ids = tokenizer.encode(
        caption,
        add_special_tokens=False,
    )

    token_ids = token_ids[:max_len - 2]

    token_ids = (
        [tokenizer.bos_token_id]
        + token_ids
        + [tokenizer.eos_token_id]
    )

    actual_length = len(token_ids)

    attention_mask = [1] * actual_length

    padding_length = max_len - actual_length

    if padding_length > 0:
        token_ids.extend(
            [tokenizer.pad_token_id] * padding_length
        )

        attention_mask.extend(
            [0] * padding_length
        )

    assert len(token_ids) == max_len
    assert len(attention_mask) == max_len

    return {
        "input_ids": token_ids,
        "attention_mask": attention_mask,
        "length": actual_length,
    }


def create_sequence_pair(encoded_caption):
    """
    Create decoder input and target sequences.

    Example:
        Full sequence: [BOS, A, dog, EOS, PAD, PAD]

        Decoder input:
            [BOS, A, dog, EOS, PAD]

        Target:
            [A, dog, EOS, PAD, PAD]

        Loss mask:
            [1, 1, 1, 0, 0]

    EOS is included in loss.
    PAD positions are ignored in loss.
    """

    input_ids = encoded_caption["input_ids"]
    attention_mask = encoded_caption["attention_mask"]

    if len(input_ids) != len(attention_mask):
        raise ValueError(
            "input_ids và attention_mask phải cùng độ dài."
        )

    if len(input_ids) < 2:
        raise ValueError(
            "Sequence cần ít nhất hai token để tạo input-target."
        )

    decoder_input_ids = input_ids[:-1]
    target_ids = input_ids[1:]

    loss_mask = attention_mask[1:]

    return {
        "decoder_input_ids": decoder_input_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
    }


def decode_caption(
    token_ids,
    tokenizer=None,
    skip_special_tokens=True,
):
    """
    Decode GPT-2 token IDs into a readable caption.

    Skip the initial BOS token.
    Stop at the next EOS token.
    """

    if tokenizer is None:
        tokenizer = load_gpt2_tokenizer()

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]

    decoded_ids = []

    for position, token_id in enumerate(token_ids):
        token_id = int(token_id)

        if position == 0 and token_id == tokenizer.bos_token_id:
            continue

        if token_id == tokenizer.eos_token_id:
            break

        decoded_ids.append(token_id)

    return tokenizer.decode(
        decoded_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    ).strip()
