import json
import os
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
        max_samples=None,
        use_cache=True,
    ):
        super().__init__()

        if split not in {"train", "val", "test"}:
            raise ValueError("split phải là 'train', 'val' hoặc 'test'")

        if max_len < 2:
            raise ValueError("max_len phải >= 2")

        self.csv_path = Path(csv_path)
        self.split_path = Path(split_path)
        self.feature_dir = Path(feature_dir)
        self.split = split
        self.max_len = max_len
        self.max_samples = max_samples
        self.use_cache = use_cache and max_samples is None

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
            raise ValueError("expected_feature_shape phải gồm 2 chiều dương")

        self._feature_shape = self.expected_feature_shape

        for path, name in [
            (self.csv_path, "CSV"),
            (self.split_path, "split"),
            (self.feature_dir, "feature"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Không tìm thấy {name}: {path}")

        with open(self.split_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

        if split not in splits:
            raise ValueError(f"Không tìm thấy split '{split}' trong JSON")

        self.image_names = set(splits[split])

        if not self.image_names:
            raise ValueError(f"Split '{split}' không có ảnh")

        for other, names in splits.items():
            if other != split and isinstance(names, list):
                overlap = self.image_names & set(names)
                if overlap:
                    raise ValueError(
                        f"Split '{split}' và '{other}' có {len(overlap)} ảnh trùng nhau"
                    )

        df = load_captions(self.csv_path)
        df = df[df["image_name"].isin(self.image_names)].copy()
        df = df[df["comment"].str.strip() != ""].copy()
        df = df.drop_duplicates(
            subset=["image_name", "comment"], keep="first"
        ).reset_index(drop=True)

        if df.empty:
            raise ValueError(f"Không có caption thuộc split '{split}'")

        if max_samples is not None:
            df = df.iloc[: int(max_samples)].copy()

        self.records = list(
            df[["image_name", "comment"]].itertuples(index=False, name=None)
        )

        self.tokenizer = load_gpt2_tokenizer()

        self._preload_features()
        self._preload_captions()

    def _read_feature(self, image_name):
        feature_path = self.feature_dir / f"{Path(image_name).stem}.npy"

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Thiếu feature cho ảnh {image_name}: {feature_path}"
            )

        array = np.load(feature_path)

        if array.ndim != 2:
            raise ValueError(
                f"Feature {image_name} phải có 2 chiều, "
                f"nhưng nhận được shape {array.shape}"
            )

        if not np.isfinite(array).all():
            raise ValueError(f"Feature {image_name} chứa NaN hoặc Inf")

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
                f"không đồng nhất với feature trước đó {self._feature_shape}"
            )

        return array

    def _cache_paths(self):
        base = self.feature_dir.parent / f"{self.feature_dir.name}_{self.split}_cache"
        return Path(str(base) + "_f16.npy"), Path(str(base) + "_names.json")

    def _try_load_cache(self, image_list):
        data_path, names_path = self._cache_paths()
        if not (data_path.exists() and names_path.exists()):
            return None
        try:
            with open(names_path, "r", encoding="utf-8") as f:
                cached_names = json.load(f)
            if cached_names != image_list:
                print(f"[{self.split}] Cache không khớp danh sách ảnh, đọc lại.", flush=True)
                return None
            features = np.load(data_path)
            if (
                features.dtype != np.float16
                or features.ndim != 3
                or features.shape[0] != len(image_list)
            ):
                return None
            if self.expected_feature_shape is not None and tuple(features.shape[1:]) != self.expected_feature_shape:
                return None
            return features
        except Exception as e:
            print(f"[{self.split}] Không đọc được cache ({e}), đọc lại.", flush=True)
            return None

    def _save_cache(self, image_list, features):
        data_path, names_path = self._cache_paths()
        try:
            tmp_data = data_path.with_name(data_path.stem + "_tmp.npy")
            np.save(tmp_data, features)
            os.replace(tmp_data, data_path)
            with open(names_path, "w", encoding="utf-8") as f:
                json.dump(image_list, f)
            print(f"[{self.split}] Đã lưu cache gộp: {data_path}", flush=True)
        except Exception as e:
            print(f"[{self.split}] Không lưu được cache ({e}), bỏ qua.", flush=True)

    def _preload_features(self):
        self.image_index = {}
        image_list = []
        for image_name, _ in self.records:
            if image_name not in self.image_index:
                self.image_index[image_name] = len(image_list)
                image_list.append(image_name)

        total = len(image_list)

        if self.use_cache:
            cached = self._try_load_cache(image_list)
            if cached is not None:
                self.features = cached
                self._feature_shape = tuple(cached.shape[1:])
                print(f"[{self.split}] Nạp feature từ cache gộp: {total} ảnh.", flush=True)
                return

        print(f"[{self.split}] Nạp sẵn feature của {total} ảnh vào RAM...", flush=True)

        self.features = None
        for i in range(total):
            array = self._read_feature(image_list[i])
            if self.features is None:
                self.features = np.zeros(
                    (total, array.shape[0], array.shape[1]), dtype=np.float16
                )
            self.features[i] = array
            if (i + 1) % 5000 == 0:
                print(f"  đã nạp {i + 1}/{total} ảnh", flush=True)

        if not np.isfinite(self.features).all():
            raise ValueError("Feature bị tràn số khi chuyển sang float16")

        if self.use_cache:
            self._save_cache(image_list, self.features)

    def _preload_captions(self):
        total = len(self.records)
        length = self.max_len - 1
        print(f"[{self.split}] Mã hóa sẵn {total} caption...", flush=True)

        self.record_image = np.zeros(total, dtype=np.int64)
        self.decoder_input_ids = np.zeros((total, length), dtype=np.int64)
        self.target_ids = np.zeros((total, length), dtype=np.int64)
        self.attention_mask = np.zeros((total, length), dtype=np.int64)
        self.loss_mask = np.zeros((total, length), dtype=np.int64)

        for i in range(total):
            image_name, caption = self.records[i]

            encoded = encode_caption(
                caption, tokenizer=self.tokenizer, max_len=self.max_len
            )
            sequence = create_sequence_pair(encoded)

            self.record_image[i] = self.image_index[image_name]
            self.decoder_input_ids[i] = sequence["decoder_input_ids"]
            self.target_ids[i] = sequence["target_ids"]
            self.attention_mask[i] = encoded["attention_mask"][:-1]
            self.loss_mask[i] = sequence["loss_mask"]

            if (i + 1) % 50000 == 0:
                print(f"  đã mã hóa {i + 1}/{total} caption", flush=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        feature = torch.from_numpy(self.features[self.record_image[index]])

        return {
            "image_features": feature,
            "decoder_input_ids": torch.from_numpy(self.decoder_input_ids[index]),
            "target_ids": torch.from_numpy(self.target_ids[index]),
            "attention_mask": torch.from_numpy(self.attention_mask[index]),
            "loss_mask": torch.from_numpy(self.loss_mask[index]),
        }