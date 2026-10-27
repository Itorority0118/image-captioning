import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.text_preprocess import encode_caption, load_captions


class Flickr30kDataset(Dataset):
    """
    Dataset Flickr30k cho Image Captioning.

    Mỗi phần tử gồm:
        image_features: Tensor (49, 576)
        caption_ids:    Tensor (max_len,)
    """

    def __init__(
        self,
        csv_path,
        split_path,
        feature_dir,
        split="train",
        vocab=None,
        max_len=40,
        min_freq=2,
    ):
        super().__init__()

        if split not in {"train", "val", "test"}:
            raise ValueError(
                "split phải là 'train', 'val' hoặc 'test'"
            )

        self.csv_path = Path(csv_path)
        self.split_path = Path(split_path)
        self.feature_dir = Path(feature_dir)
        self.split = split
        self.max_len = max_len

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy CSV: {self.csv_path}"
            )

        if not self.split_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy split: {self.split_path}"
            )

        if not self.feature_dir.exists():
            raise FileNotFoundError(
                f"Không tìm thấy feature: {self.feature_dir}"
            )

        # Đọc danh sách ảnh train, validation và test.
        with open(self.split_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

        if split not in splits:
            raise ValueError(
                f"Không tìm thấy split '{split}' trong JSON"
            )

        self.image_names = set(splits[split])

        # Đọc caption Flickr30k.
        df = load_captions(self.csv_path)

        # Chỉ giữ caption thuộc tập ảnh hiện tại
        df = df[df["image_name"].isin(self.image_names)].copy()

        # Loại caption rỗng và bản ghi trùng trong cùng một ảnh
        df = df[df["comment"].str.strip() != ""].copy()

        df = df.drop_duplicates(
            subset=["image_name", "comment"],
            keep="first"
        ).reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"Không có caption nào thuộc tập {split}"
            )

        # Mỗi ảnh có thể có nhiều caption.
        self.records = list(
            df[["image_name", "comment"]]
            .itertuples(index=False, name=None)
        )

        # Chỉ xây dựng vocabulary từ caption của train.
        if vocab is None:
            if split != "train":
                raise ValueError(
                    "Val/test phải dùng vocabulary được tạo từ train."
                )

            from src.data.text_preprocess import build_vocab

            self.vocab = build_vocab(
                df["comment"].tolist(),
                min_freq=min_freq,
            )
        else:
            self.vocab = vocab

        self._feature_cache = {}

    def __len__(self):
        return len(self.records)

    def _load_feature(self, image_name):
        # Giữ cache một feature gần nhất để giảm đọc file lặp lại.
        if image_name in self._feature_cache:
            return self._feature_cache[image_name]

        feature_path = (
            self.feature_dir
            / f"{Path(image_name).stem}.npy"
        )

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Thiếu feature cho ảnh {image_name}: "
                f"{feature_path}"
            )

        array = np.load(feature_path)

        if array.shape != (49, 576):
            raise ValueError(
                f"Feature {image_name} có shape {array.shape}, "
                "mong đợi (49, 576)"
            )

        if not np.isfinite(array).all():
            raise ValueError(
                f"Feature {image_name} chứa NaN hoặc Inf"
            )

        # Chuyển float16 sang float32 cho quá trình huấn luyện.
        feature = torch.from_numpy(
            array.astype(np.float32)
        )

        self._feature_cache.clear()
        self._feature_cache[image_name] = feature

        return feature

    def __getitem__(self, index):
        image_name, caption = self.records[index]

        image_features = self._load_feature(image_name)

        caption_ids = encode_caption(
            caption,
            self.vocab,
            max_len=self.max_len,
        )

        caption_ids = torch.tensor(
            caption_ids,
            dtype=torch.long,
        )

        return image_features, caption_ids
