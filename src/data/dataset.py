
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.text_preprocess import (
    MAX_LEN,
    create_sequence_pair,
    encode_caption,
    load_captions,
    load_gpt2_tokenizer,
)


class Flickr30kDataset(Dataset):

    def __init__(
        self,
        csv_path,
        split_path,
        feature_dir,
        split="train",
        max_len=MAX_LEN,
        expected_feature_shape=None,
    ):
        super().__init__()

        if split not in {"train", "val", "test"}:
            raise ValueError(
                "split phải là 'train', 'val' hoặc 'test'"
            )

        if max_len < 2:
            raise ValueError("max_len phải >= 2")

        self.csv_path = Path(csv_path)
        self.split_path = Path(split_path)
        self.feature_dir = Path(feature_dir)
        self.split = split
        self.max_len = max_len

        self.expected_feature_shape = (
            tuple(expected_feature_shape)
            if expected_feature_shape is not None
            else None
        )

        if (
            self.expected_feature_shape is not None
            and (
                len(self.expected_feature_shape) != 2
                or any(dim <= 0 for dim in self.expected_feature_shape)
            )
        ):
            raise ValueError(
                "expected_feature_shape phải gồm 2 chiều dương"
            )

        self._feature_shape = self.expected_feature_shape
        self._feature_cache = {}

        for path, name in [
            (self.csv_path, "CSV"),
            (self.split_path, "split"),
            (self.feature_dir, "feature"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy {name}: {path}"
                )

        with open(self.split_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

        if split not in splits:
            raise ValueError(
                f"Không tìm thấy split '{split}' trong JSON"
            )

        self.image_names = set(splits[split])

        if not self.image_names:
            raise ValueError(
                f"Split '{split}' không có ảnh"
            )

        df = load_captions(self.csv_path)

        df = df[
            df["image_name"].isin(self.image_names)
        ].copy()

        df = df[
            df["comment"].str.strip() != ""
        ].copy()

        df = df.drop_duplicates(
            subset=["image_name", "comment"],
            keep="first",
        ).reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"Không có caption thuộc split '{split}'"
            )

        self.records = list(
            df[["image_name", "comment"]]
            .itertuples(index=False, name=None)
        )

        self.tokenizer = load_gpt2_tokenizer()

    def __len__(self):
        return len(self.records)

    def _load_feature(self, image_name):
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

        if array.ndim != 2:
            raise ValueError(
                f"Feature {image_name} phải có 2 chiều, "
                f"nhưng nhận được shape {array.shape}"
            )

        if not np.isfinite(array).all():
            raise ValueError(
                f"Feature {image_name} chứa NaN hoặc Inf"
            )

        current_shape = tuple(array.shape)

        if self.expected_feature_shape is not None:
            if current_shape != self.expected_feature_shape:
                raise ValueError(
                    f"Feature {image_name} có shape {current_shape}, "
                    f"mong đợi {self.expected_feature_shape}"
                )

        if self._feature_shape is None:
            self._feature_shape = current_shape

        elif current_shape != self._feature_shape:
            raise ValueError(
                f"Feature {image_name} có shape {current_shape}, "
                f"không đồng nhất với feature trước đó "
                f"{self._feature_shape}"
            )

        feature = torch.from_numpy(
            array.astype(np.float32)
        )

        self._feature_cache.clear()
        self._feature_cache[image_name] = feature

        return feature

    def __getitem__(self, index):
        image_name, caption = self.records[index]

        image_features = self._load_feature(image_name)

        encoded = encode_caption(
            caption,
            tokenizer=self.tokenizer,
            max_len=self.max_len,
        )

        sequence = create_sequence_pair(encoded)

        return {
            "image_features": image_features,
            "decoder_input_ids": torch.tensor(
                sequence["decoder_input_ids"],
                dtype=torch.long,
            ),
            "target_ids": torch.tensor(
                sequence["target_ids"],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                encoded["attention_mask"][:-1],
                dtype=torch.long,
            ),
            "loss_mask": torch.tensor(
                sequence["loss_mask"],
                dtype=torch.long,
            ),
        }