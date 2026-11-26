"""Evaluate Flickr30k image captioning checkpoints with four search methods.

Example:
    python -m src.eval.evaluate \
        --csv_path data/results.csv \
        --features_root outputs/features \
        --checkpoint efficientnet_b0_gpt2=/path/to/efficientnet_best.pt

Repeat ``--checkpoint MODEL=PATH`` for checkpoints that are not in the
standard ``<checkpoint_root>/<model>/best.pt`` locations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from src.data.text_preprocess import GPT2_VOCAB_SIZE, load_captions, load_gpt2_tokenizer
from src.decoding.beam import beam_search
from src.decoding.con_search import contrastive_search
from src.decoding.diverse_beam import diverse_beam_search
from src.decoding.greedy import greedy_decode
from src.models.decoder_gpt2 import GPT2Decoder
from src.models.encoders import EXPECTED_SHAPES
from src.paths import DATA_PATH, WORK_PATH


@dataclass(frozen=True)
class ModelSpec:
    name: str
    encoder: str
    decoder: str


MODEL_SPECS = (
    ModelSpec("efficientnet_b0_gpt2", "efficientnet_b0", "gpt2"),
    ModelSpec("mobilenetv3_lstm", "mobilenetv3", "lstm"),
    ModelSpec("tinyclip_gpt2", "tinyclip", "gpt2"),
    ModelSpec("deit_tiny_gpt2", "deit_tiny", "gpt2"),
)
MODEL_NAMES = {spec.name for spec in MODEL_SPECS}
REPORT_SCORES = ("BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "SPICE")


def _parse_checkpoint_override(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or name not in MODEL_NAMES or not path:
        raise argparse.ArgumentTypeError(
            "checkpoint phải có dạng MODEL=PATH, với MODEL là một trong: "
            + ", ".join(sorted(MODEL_NAMES))
        )
    return name, Path(path).expanduser()


def _build_model(spec: ModelSpec, device: torch.device) -> torch.nn.Module:
    feature_dim = EXPECTED_SHAPES[spec.encoder][1]
    if spec.decoder == "lstm":
        from src.models.decoder_lstm import LSTMDecoder

        return LSTMDecoder(feature_dim, GPT2_VOCAB_SIZE).to(device)
    return GPT2Decoder(feature_dim).to(device)


def _load_checkpoint(
    path: Path, spec: ModelSpec, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint {spec.name}: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint {path} không chứa dictionary hợp lệ")

    encoder = checkpoint.get("encoder")
    decoder = checkpoint.get("decoder")
    if encoder is not None and encoder != spec.encoder:
        raise ValueError(
            f"Checkpoint {path} ghi encoder={encoder}, cần {spec.encoder}"
        )
    if decoder is not None and decoder != spec.decoder:
        raise ValueError(
            f"Checkpoint {path} ghi decoder={decoder}, cần {spec.decoder}"
        )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Không tìm thấy model_state_dict trong {path}")

    model = _build_model(spec, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, {
        "checkpoint": str(path.resolve()),
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_mtime_ns": path.stat().st_mtime_ns,
        "epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
        "best_val": checkpoint.get("best_val"),
    }


def _load_references(csv_path: Path, split_path: Path) -> tuple[list[str], dict[str, list[str]]]:
    if not split_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy split JSON: {split_path}")
    with split_path.open("r", encoding="utf-8") as file:
        splits = json.load(file)
    if "test" not in splits:
        raise ValueError(f"Split JSON không có khóa 'test': {split_path}")

    frame = load_captions(csv_path)
    references: dict[str, list[str]] = {}
    for image_name, caption in frame[["image_name", "comment"]].itertuples(
        index=False, name=None
    ):
        references.setdefault(str(image_name), []).append(str(caption))

    test_names = list(dict.fromkeys(str(name) for name in splits["test"]))
    test_names = [name for name in test_names if name in references]
    if not test_names:
        raise ValueError("Không tìm thấy ảnh test nào có caption trong CSV")
    return test_names, references


def _select_images(
    test_names: list[str],
    features_root: Path,
    sample_size: int,
    seed: int,
) -> list[str]:
    # Dùng giao của đủ bốn encoder để mọi mô hình được chấm trên cùng ảnh.
    common = []
    for image_name in test_names:
        stem = Path(image_name).stem
        if all(
            (features_root / spec.encoder / f"{stem}.npy").is_file()
            for spec in MODEL_SPECS
        ):
            common.append(image_name)

    if not common:
        raise FileNotFoundError(
            f"Không có ảnh test nào đủ feature của cả bốn mô hình dưới {features_root}"
        )
    if sample_size < 0:
        raise ValueError("sample_size phải >= 0; dùng 0 để lấy toàn bộ test set")
    if sample_size and sample_size > len(common):
        raise ValueError(
            f"Chỉ có {len(common)} ảnh test đủ feature, không đủ sample_size={sample_size}"
        )

    names = sorted(common)
    if sample_size:
        names = sorted(random.Random(seed).sample(names, sample_size))
    return names


def _load_feature(path: Path, expected_shape: tuple[int, int]) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy feature: {path}")
    array = np.load(path, allow_pickle=False)
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            f"Feature {path.name} có shape {tuple(array.shape)}, "
            f"mong đợi {expected_shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"Feature có NaN hoặc Inf: {path}")
    return torch.from_numpy(array.astype(np.float32, copy=False))


@lru_cache(maxsize=1)
def _java_major_version() -> int:
    version = subprocess.run(
        ["java", "-version"], capture_output=True, text=True, timeout=15, check=True
    )
    match = re.search(r'version "(?:1\.)?(\d+)', version.stderr + version.stdout)
    if not match:
        raise RuntimeError(f"Could not determine Java version: {version.stderr}")
    return int(match[1])


def _spice_score(scorer, gts, res) -> float:
    if _java_major_version() < 15:
        return float(scorer.compute_score(gts, res)[0])

    # Use the original SPICE scorer and write its JSON without Nashorn.
    from pycocoevalcap.spice import spice

    spice_dir = Path(spice.__file__).resolve().parent
    launcher = Path(__file__).with_name("SpiceJson.java").resolve()
    entries = [
        {"image_id": key, "test": res[key][0], "refs": gts[key]}
        for key in gts
    ]
    with tempfile.TemporaryDirectory(prefix="caption-spice-") as folder:
        input_path = Path(folder) / "input.json"
        output_path = Path(folder) / "output.json"
        input_path.write_text(json.dumps(entries), encoding="utf-8")
        cache_dir = spice_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        packages = (
            "java.lang", "java.math", "java.util", "java.util.concurrent",
            "java.io", "java.net", "java.text", "java.time", "java.lang.reflect",
        )
        subprocess.run(
            [
                "java", "-Xmx4G",
                *(f"--add-opens=java.base/{package}=ALL-UNNAMED" for package in packages),
                "-cp", os.pathsep.join((spice.SPICE_JAR, "lib/*")),
                str(launcher), str(output_path), str(input_path),
                "-cache", str(cache_dir), "-subset", "-silent",
            ],
            cwd=spice_dir,
            check=True,
        )
        results = json.loads(output_path.read_text(encoding="utf-8"))
    if len(results) != len(entries):
        raise ValueError(f"SPICE returned {len(results)} rows for {len(entries)} images")
    return float(np.mean([scorer.float_convert(row["scores"]["All"]["f"]) for row in results]))


def _caption_metrics(
    image_names: list[str],
    references: dict[str, list[str]],
    predictions: list[str],
) -> dict[str, float]:
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.spice.spice import Spice
    except ImportError as error:
        raise RuntimeError(
            "Thiếu pycocoevalcap; cài thư viện bằng `pip install -r requirements.txt`."
        ) from error

    if len(predictions) != len(image_names):
        raise ValueError("One generated caption is required for each test image")
    gts = {str(i): references[name] for i, name in enumerate(image_names)}
    res = {str(i): [caption] for i, caption in enumerate(predictions)}
    bleu_scores, _ = Bleu(4).compute_score(gts, res)
    meteor_scorer = Meteor()
    try:
        meteor_score, _ = meteor_scorer.compute_score(gts, res)
    finally:
        close = getattr(meteor_scorer, "close", None)
        if close is not None:
            close()
    rouge_score, _ = Rouge().compute_score(gts, res)
    cider_score, _ = Cider().compute_score(gts, res)
    spice_score = _spice_score(Spice(), gts, res)

    return {
        **{f"BLEU-{index + 1}": float(score) for index, score in enumerate(bleu_scores)},
        "METEOR": float(meteor_score),
        "ROUGE-L": float(rouge_score),
        "CIDEr": float(cider_score),
        "SPICE": float(spice_score),
    }


def _resolve_checkpoint_paths(
    checkpoint_root: Path, overrides: list[tuple[str, Path]]
) -> dict[str, Path]:
    paths = {
        spec.name: checkpoint_root / spec.name / "best.pt" for spec in MODEL_SPECS
    }
    for name, path in overrides:
        paths[name] = path
    return paths


def _write_tables(output_dir: Path, rows: list[dict], diverse_groups: int) -> Path:
    metrics_path = output_dir / "metrics.csv"
    report_fields = [
        "model", "method", "group", "test_images", "checkpoint_epoch", *REPORT_SCORES,
    ]
    comparison = []
    groups_by_model: dict[str, list[dict]] = {}
    for row in rows:
        if row["method"] != "diverse_beam":
            comparison.append(row)
            continue
        groups = groups_by_model.setdefault(row["model"], [])
        groups.append(row)
        if len(groups) == diverse_groups:
            comparison.append({
                **row,
                "group": "mean_all_groups",
                **{key: float(np.mean([group[key] for group in groups])) for key in REPORT_SCORES},
            })
    for path, fields, data in (
        (metrics_path, list(rows[0]), rows),
        (output_dir / "comparison_6_scores.csv", report_fields, comparison),
    ):
        temp_path = path.with_suffix(".tmp")
        with temp_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
        temp_path.replace(path)
    return metrics_path


def evaluate(args: argparse.Namespace) -> tuple[list[dict], Path]:
    csv_path = Path(args.csv_path).expanduser()
    split_path = Path(args.splits).expanduser()
    features_root = Path(args.features_root).expanduser()
    checkpoint_root = Path(args.checkpoint_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if args.reuse_captions_from and args.reuse_captions_from.resolve() == output_dir.resolve():
        raise ValueError("Reused captions and evaluation output must use different directories")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Đã chọn CUDA nhưng runtime không có GPU khả dụng")

    test_names, references = _load_references(csv_path, split_path)
    image_names = _select_images(
        test_names, features_root, args.sample_size, args.seed
    )
    tokenizer = load_gpt2_tokenizer()
    checkpoint_paths = _resolve_checkpoint_paths(
        checkpoint_root, args.checkpoint
    )
    _java_major_version()

    print(
        f"Thiết bị: {device} | Ảnh đánh giá: {len(image_names)} "
        f"| Caption tham chiếu trung bình: "
        f"{np.mean([len(references[name]) for name in image_names]):.2f}",
        flush=True,
    )

    rows: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in MODEL_SPECS:
        model, checkpoint_info = _load_checkpoint(
            checkpoint_paths[spec.name], spec, device
        )
        print(
            f"\n{spec.name}: checkpoint epoch={checkpoint_info['epoch']} "
            f"val_loss={checkpoint_info['val_loss']}",
            flush=True,
        )

        beam_predictions: list[str] = []
        greedy_predictions: list[str] = []
        contrastive_predictions: list[str] = []
        greedy_seconds = contrastive_seconds = 0.0
        diverse_predictions: list[list[str]] = [
            [] for _ in range(args.diverse_groups)
        ]
        beam_seconds = 0.0
        diverse_seconds = 0.0

        caption_path = output_dir / f"{spec.name}_captions.csv"
        groups = [f"diverse_beam_{i + 1}" for i in range(args.diverse_groups)]
        columns = ["image", "references", "greedy", "beam", "contrastive", *groups]
        reused: list[dict[str, str]] = []
        if args.reuse_captions_from:
            source = Path(args.reuse_captions_from) / caption_path.name
            if source.is_file():
                with source.open(newline="", encoding="utf-8-sig") as file:
                    reader = csv.DictReader(file)
                    if reader.fieldnames != columns:
                        raise ValueError(f"Unexpected caption columns in {source}")
                    reused = list(reader)
                if [row["image"] for row in reused] != image_names[:len(reused)] or any(
                    any(row[column] is None for column in columns)
                    or row["references"].splitlines() != references[row["image"]]
                    for row in reused
                ):
                    raise ValueError(f"Captions in {source} do not match this evaluation split")
                for row in reused:
                    greedy_predictions.append(row["greedy"])
                    beam_predictions.append(row["beam"])
                    contrastive_predictions.append(row["contrastive"])
                    for group, name in zip(diverse_predictions, groups):
                        group.append(row[name])
                print(f"Reused {len(reused)} captions from {source}", flush=True)

        with caption_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(reused)

        for index, image_name in enumerate(image_names[len(reused):], start=len(reused) + 1):
            feature_path = features_root / spec.encoder / f"{Path(image_name).stem}.npy"
            feature = _load_feature(feature_path, EXPECTED_SHAPES[spec.encoder])

            start = time.perf_counter()
            greedy_predictions.append(greedy_decode(model, feature, max_len=args.max_len))
            greedy_seconds += time.perf_counter() - start

            start = time.perf_counter()
            contrastive_predictions.append(contrastive_search(model, feature, max_len=args.max_len))
            contrastive_seconds += time.perf_counter() - start

            start = time.perf_counter()
            beam_caption = beam_search(
                model,
                feature,
                tokenizer=tokenizer,
                beam_size=args.beam_size,
                max_len=args.max_len,
                length_penalty=args.length_penalty,
            )
            beam_seconds += time.perf_counter() - start
            beam_predictions.append(beam_caption)

            start = time.perf_counter()
            group_captions = diverse_beam_search(
                model,
                feature,
                tokenizer=tokenizer,
                beam_size=args.diverse_beam_size,
                num_groups=args.diverse_groups,
                diversity_strength=args.diversity_strength,
                max_len=args.max_len,
                length_penalty=args.length_penalty,
            )
            diverse_seconds += time.perf_counter() - start
            for group_index, caption in enumerate(group_captions):
                diverse_predictions[group_index].append(caption)

            with caption_path.open("a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([image_name, "\n".join(references[image_name]),
                                 greedy_predictions[-1], beam_caption,
                                 contrastive_predictions[-1], *group_captions])

            if index % 10 == 0 or index == len(image_names):
                print(
                    f"  {index}/{len(image_names)} ảnh đã xử lý",
                    flush=True,
                )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        metric_sets = [
            ("greedy", "", greedy_predictions, greedy_seconds),
            ("beam", "", beam_predictions, beam_seconds),
            ("contrastive", "", contrastive_predictions, contrastive_seconds),
        ]
        metric_sets.extend(
            (
                "diverse_beam",
                f"group_{group_index + 1}",
                predictions,
                diverse_seconds,
            )
            for group_index, predictions in enumerate(diverse_predictions)
        )
        for method, group, predictions, elapsed_seconds in metric_sets:
            print(f"  Chấm {method} {group}".rstrip(), flush=True)
            scores = _caption_metrics(image_names, references, predictions)
            row = {
                "model": spec.name,
                "encoder": spec.encoder,
                "decoder": spec.decoder,
                "method": method,
                "group": group,
                "test_images": len(image_names),
                "checkpoint_epoch": checkpoint_info["epoch"],
                "checkpoint_val_loss": checkpoint_info["val_loss"],
                "seconds": None if reused else elapsed_seconds,
                **scores,
            }
            rows.append(row)
            metrics_path = _write_tables(output_dir, rows, args.diverse_groups)

        prediction_data = {
            "model": spec.name,
            "encoder": spec.encoder,
            "decoder": spec.decoder,
            "checkpoint": checkpoint_info,
            "device": str(device),
            "seed": args.seed,
            "test_images": len(image_names),
            "beam_size": args.beam_size,
            "diverse_beam_size": args.diverse_beam_size,
            "diverse_groups": args.diverse_groups,
            "diversity_strength": args.diversity_strength,
            "predictions": [
                {
                    "image_name": image_name,
                    "references": references[image_name],
                    "greedy_search": greedy_predictions[index],
                    "contrastive_search": contrastive_predictions[index],
                    "beam_search": beam_predictions[index],
                    "diverse_beam_search": [
                        group[index] for group in diverse_predictions
                    ],
                }
                for index, image_name in enumerate(image_names)
            ],
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = output_dir / f"{spec.name}_predictions.json"
        prediction_path.write_text(
            json.dumps(prediction_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Đã lưu dự đoán: {prediction_path}", flush=True)

    print(f"\nĐã lưu bảng chỉ số: {metrics_path}", flush=True)
    print("\nKết quả:")
    for row in rows:
        print(
            f"{row['model']:24s} {row['method']:13s} {row['group']:8s} "
            f"BLEU-4={row['BLEU-4']:.4f} METEOR={row['METEOR']:.4f} "
            f"ROUGE-L={row['ROUGE-L']:.4f} CIDEr={row['CIDEr']:.4f} "
            f"SPICE={row['SPICE']:.4f}"
        )
    return rows, metrics_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Đánh giá bốn mô hình Flickr30k với bốn thuật toán sinh caption."
    )
    parser.add_argument("--csv_path", default=str(DATA_PATH / "results.csv"))
    parser.add_argument("--splits", default="configs/splits.json")
    parser.add_argument("--features_root", default=str(WORK_PATH / "features"))
    parser.add_argument("--checkpoint_root", default=str(WORK_PATH / "checkpoints"))
    parser.add_argument("--checkpoint", action="append", type=_parse_checkpoint_override, default=[])
    parser.add_argument("--output_dir", default=str(WORK_PATH / "evaluation"))
    parser.add_argument(
        "--reuse_captions_from", type=Path,
        help="Reuse captions from a run with the same checkpoints and decoding settings.",
    )
    parser.add_argument("--sample_size", type=int, default=1000,
                        help="Số ảnh chọn từ test split; 0 dùng toàn bộ test set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--diverse_beam_size", type=int, default=6)
    parser.add_argument("--diverse_groups", type=int, default=3)
    parser.add_argument("--diversity_strength", type=float, default=0.5)
    parser.add_argument("--max_len", type=int, default=40)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.beam_size < 1:
        raise ValueError("beam_size phải >= 1")
    if args.diverse_groups < 1 or args.diverse_beam_size < args.diverse_groups:
        raise ValueError("diverse_beam_size phải >= diverse_groups >= 1")
    if args.diverse_beam_size % args.diverse_groups:
        raise ValueError("diverse_beam_size phải chia hết cho diverse_groups")
    evaluate(args)


if __name__ == "__main__":
    main()
