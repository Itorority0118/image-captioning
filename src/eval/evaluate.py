import argparse
import csv
import importlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from src.data.text_preprocess import GPT2_VOCAB_SIZE, load_captions, load_gpt2_tokenizer
from src.decoding.greedy import greedy_decode
from src.decoding.beam import beam_search
from src.decoding.diverse_beam import diverse_beam_search
from src.models.decoder_gpt2 import GPT2Decoder
from src.models.encoders import EXPECTED_SHAPES
from src.paths import DATA_PATH, WORK_PATH

MODELS = {
    "efficientnet_b0_gpt2": ("efficientnet_b0", "gpt2"),
    "mobilenetv3_lstm": ("mobilenetv3", "lstm"),
    "tinyclip_gpt2": ("tinyclip", "gpt2"),
    "deit_tiny_gpt2": ("deit_tiny", "gpt2"),
}
METHODS = ["greedy", "beam", "contrastive", "diverse_beam"]


def parse_checkpoint_override(value):
    parts = value.split("=", 1)
    if len(parts) != 2 or parts[0] not in MODELS or parts[1] == "":
        raise argparse.ArgumentTypeError(
            "checkpoint phải có dạng MODEL=PATH, với MODEL là một trong: "
            + ", ".join(sorted(MODELS.keys()))
        )
    return parts[0], Path(parts[1]).expanduser()


def load_contrastive_function():
    for module_name in ["contrastive", "con_search"]:
        try:
            module = importlib.import_module("src.decoding." + module_name)
            return module.contrastive_search
        except ModuleNotFoundError:
            pass
    return None


def build_model(model_name, device):
    encoder_name, decoder_name = MODELS[model_name]
    feature_dim = EXPECTED_SHAPES[encoder_name][1]
    if decoder_name == "lstm":
        from src.models.decoder_lstm import LSTMDecoder

        return LSTMDecoder(feature_dim, GPT2_VOCAB_SIZE).to(device)
    return GPT2Decoder(feature_dim).to(device)


def load_checkpoint(path, model_name, device):
    if not path.is_file():
        raise FileNotFoundError("Không tìm thấy checkpoint " + model_name + ": " + str(path))

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint " + str(path) + " không chứa dictionary hợp lệ")

    encoder_name, decoder_name = MODELS[model_name]
    saved_encoder = checkpoint.get("encoder")
    saved_decoder = checkpoint.get("decoder")
    if saved_encoder is not None and saved_encoder != encoder_name:
        raise ValueError("Checkpoint " + str(path) + " ghi encoder=" + str(saved_encoder) + ", cần " + encoder_name)
    if saved_decoder is not None and saved_decoder != decoder_name:
        raise ValueError("Checkpoint " + str(path) + " ghi decoder=" + str(saved_decoder) + ", cần " + decoder_name)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = build_model(model_name, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    info = {
        "checkpoint": str(path),
        "epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
        "best_val": checkpoint.get("best_val"),
    }
    return model, info


def load_references(csv_path, split_path):
    if not split_path.is_file():
        raise FileNotFoundError("Không tìm thấy split JSON: " + str(split_path))
    f = open(split_path, "r", encoding="utf-8")
    splits = json.load(f)
    f.close()
    if "test" not in splits:
        raise ValueError("Split JSON không có khóa 'test': " + str(split_path))

    frame = load_captions(csv_path)
    references = {}
    for image_name, caption in frame[["image_name", "comment"]].itertuples(index=False, name=None):
        image_name = str(image_name)
        if image_name not in references:
            references[image_name] = []
        references[image_name].append(str(caption))

    test_names = []
    seen = set()
    for name in splits["test"]:
        name = str(name)
        if name in seen:
            continue
        seen.add(name)
        if name in references:
            test_names.append(name)
    if len(test_names) == 0:
        raise ValueError("Không tìm thấy ảnh test nào có caption trong CSV")
    return test_names, references


def select_images(test_names, features_root, model_names, sample_size, seed):
    encoders = []
    for model_name in model_names:
        encoder_name = MODELS[model_name][0]
        if encoder_name not in encoders:
            encoders.append(encoder_name)

    common = []
    for image_name in test_names:
        stem = Path(image_name).stem
        has_all = True
        for encoder_name in encoders:
            if not (features_root / encoder_name / (stem + ".npy")).is_file():
                has_all = False
                break
        if has_all:
            common.append(image_name)

    if len(common) == 0:
        raise FileNotFoundError(
            "Không có ảnh test nào đủ feature của các encoder " + ", ".join(encoders) + " dưới " + str(features_root)
        )
    if sample_size < 0:
        raise ValueError("sample_size phải >= 0; dùng 0 để lấy toàn bộ test set")
    if sample_size > len(common):
        raise ValueError(
            "Chỉ có " + str(len(common)) + " ảnh test đủ feature, không đủ sample_size=" + str(sample_size)
        )

    names = sorted(common)
    if sample_size > 0:
        names = sorted(random.Random(seed).sample(names, sample_size))
    return names


def load_feature(path, expected_shape):
    if not path.is_file():
        raise FileNotFoundError("Không tìm thấy feature: " + str(path))
    array = np.load(path, allow_pickle=False)
    if tuple(array.shape) != tuple(expected_shape):
        raise ValueError(
            "Feature " + path.name + " có shape " + str(tuple(array.shape)) + ", mong đợi " + str(tuple(expected_shape))
        )
    if not np.isfinite(array).all():
        raise ValueError("Feature có NaN hoặc Inf: " + str(path))
    return torch.from_numpy(array.astype(np.float32))


def check_java():
    if shutil.which("java") is None:
        raise RuntimeError(
            "Không thấy Java. METEOR, SPICE và bộ tách từ PTB đều cần Java "
            "(trên Colab thử: !apt-get install -y default-jre-headless)."
        )


def compute_metrics(image_names, references, predictions, skip_spice):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    gts_raw = {}
    res_raw = {}
    for i in range(len(image_names)):
        key = str(i)
        gts_raw[key] = []
        for text in references[image_names[i]]:
            gts_raw[key].append({"caption": text})
        caption = predictions[i]
        if caption.strip() == "":
            caption = "."
        res_raw[key] = [{"caption": caption}]

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts_raw)
    res = tokenizer.tokenize(res_raw)

    scores = {}
    bleu_scores, _ = Bleu(4).compute_score(gts, res)
    for index in range(4):
        scores["BLEU-" + str(index + 1)] = float(bleu_scores[index])

    meteor_scorer = Meteor()
    try:
        meteor_score, _ = meteor_scorer.compute_score(gts, res)
    finally:
        close = getattr(meteor_scorer, "close", None)
        if close is not None:
            close()
    scores["METEOR"] = float(meteor_score)

    rouge_score, _ = Rouge().compute_score(gts, res)
    scores["ROUGE-L"] = float(rouge_score)

    cider_score, _ = Cider().compute_score(gts, res)
    scores["CIDEr"] = float(cider_score)

    if skip_spice:
        scores["SPICE"] = None
    else:
        from pycocoevalcap.spice.spice import Spice

        spice_score, _ = Spice().compute_score(gts, res)
        scores["SPICE"] = float(spice_score)
    return scores


def generate_captions(model, model_name, image_names, features_root, args, tokenizer, contrastive_search):
    encoder_name = MODELS[model_name][0]
    expected_shape = EXPECTED_SHAPES[encoder_name]

    predictions = {}
    seconds = {}
    for method in args.methods:
        seconds[method] = 0.0
        if method == "diverse_beam":
            predictions[method] = []
            for group_index in range(args.diverse_groups):
                predictions[method].append([])
        else:
            predictions[method] = []

    for index in range(len(image_names)):
        image_name = image_names[index]
        feature_path = features_root / encoder_name / (Path(image_name).stem + ".npy")
        feature = load_feature(feature_path, expected_shape)

        for method in args.methods:
            start = time.perf_counter()
            if method == "greedy":
                predictions[method].append(greedy_decode(model, feature, max_len=args.max_len))
            elif method == "beam":
                caption = beam_search(
                    model,
                    feature,
                    tokenizer=tokenizer,
                    beam_size=args.beam_size,
                    max_len=args.max_len,
                    length_penalty=args.length_penalty,
                )
                predictions[method].append(caption)
            elif method == "contrastive":
                caption = contrastive_search(
                    model,
                    feature,
                    max_len=args.max_len,
                    top_k=args.contrastive_top_k,
                    alpha=args.contrastive_alpha,
                )
                predictions[method].append(caption)
            else:
                groups = diverse_beam_search(
                    model,
                    feature,
                    tokenizer=tokenizer,
                    beam_size=args.diverse_beam_size,
                    num_groups=args.diverse_groups,
                    diversity_strength=args.diversity_strength,
                    max_len=args.max_len,
                    length_penalty=args.length_penalty,
                )
                for group_index in range(len(groups)):
                    predictions[method][group_index].append(groups[group_index])
            seconds[method] = seconds[method] + (time.perf_counter() - start)

        if (index + 1) % 10 == 0 or index + 1 == len(image_names):
            print("  " + str(index + 1) + "/" + str(len(image_names)) + " ảnh đã sinh caption", flush=True)

    return predictions, seconds


def save_predictions(output_dir, model_name, info, image_names, references, predictions, args, device):
    items = []
    for index in range(len(image_names)):
        item = {"image_name": image_names[index], "references": references[image_names[index]]}
        for method in args.methods:
            if method == "diverse_beam":
                groups = []
                for group in predictions[method]:
                    groups.append(group[index])
                item[method] = groups
            else:
                item[method] = predictions[method][index]
        items.append(item)

    data = {
        "model": model_name,
        "checkpoint": info,
        "device": str(device),
        "seed": args.seed,
        "test_images": len(image_names),
        "methods": args.methods,
        "settings": {
            "beam_size": args.beam_size,
            "diverse_beam_size": args.diverse_beam_size,
            "diverse_groups": args.diverse_groups,
            "diversity_strength": args.diversity_strength,
            "contrastive_top_k": args.contrastive_top_k,
            "contrastive_alpha": args.contrastive_alpha,
            "max_len": args.max_len,
            "length_penalty": args.length_penalty,
        },
        "predictions": items,
    }
    path = output_dir / (model_name + "_predictions.json")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Đã lưu dự đoán:", path, flush=True)


def evaluate(args):
    csv_path = Path(args.csv_path).expanduser()
    split_path = Path(args.splits).expanduser()
    features_root = Path(args.features_root).expanduser()
    checkpoint_root = Path(args.checkpoint_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Đã chọn CUDA nhưng runtime không có GPU khả dụng")

    check_java()

    contrastive_search = None
    if "contrastive" in args.methods:
        contrastive_search = load_contrastive_function()
        if contrastive_search is None:
            raise FileNotFoundError("Không thấy contrastive.py hoặc con_search.py trong src/decoding/")

    checkpoint_paths = {}
    for model_name in args.models:
        checkpoint_paths[model_name] = checkpoint_root / model_name / "best.pt"
    for model_name, path in args.checkpoint:
        checkpoint_paths[model_name] = path
    missing = []
    for model_name in args.models:
        if not checkpoint_paths[model_name].is_file():
            missing.append(model_name + ": " + str(checkpoint_paths[model_name]))
    if len(missing) > 0:
        raise FileNotFoundError("Thiếu checkpoint:\n  " + "\n  ".join(missing))

    test_names, references = load_references(csv_path, split_path)
    image_names = select_images(test_names, features_root, args.models, args.sample_size, args.seed)
    tokenizer = load_gpt2_tokenizer()

    total_refs = 0
    for name in image_names:
        total_refs = total_refs + len(references[name])
    print(
        "Thiết bị: " + str(device) + " | Mô hình: " + ", ".join(args.models)
        + " | Thuật toán: " + ", ".join(args.methods)
        + " | Ảnh đánh giá: " + str(len(image_names))
        + " | Caption tham chiếu trung bình: " + str(round(total_refs / len(image_names), 2)),
        flush=True,
    )
    if not args.skip_spice:
        print("Lưu ý: lần đầu chạy SPICE sẽ tải Stanford CoreNLP, cần kết nối mạng.", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for model_name in args.models:
        encoder_name, decoder_name = MODELS[model_name]
        model, info = load_checkpoint(checkpoint_paths[model_name], model_name, device)
        print("\n" + model_name + ": checkpoint epoch=" + str(info["epoch"]) + " val_loss=" + str(info["val_loss"]), flush=True)

        predictions, seconds = generate_captions(
            model, model_name, image_names, features_root, args, tokenizer, contrastive_search
        )
        save_predictions(output_dir, model_name, info, image_names, references, predictions, args, device)

        for method in args.methods:
            if method == "diverse_beam":
                prediction_sets = []
                for group_index in range(len(predictions[method])):
                    prediction_sets.append(("group_" + str(group_index + 1), predictions[method][group_index]))
            else:
                prediction_sets = [("", predictions[method])]

            for group_name, caption_list in prediction_sets:
                print("  Chấm điểm " + method + " " + group_name + "...", flush=True)
                scores = compute_metrics(image_names, references, caption_list, args.skip_spice)
                row = {
                    "model": model_name,
                    "encoder": encoder_name,
                    "decoder": decoder_name,
                    "method": method,
                    "group": group_name,
                    "test_images": len(image_names),
                    "checkpoint_epoch": info["epoch"],
                    "checkpoint_val_loss": info["val_loss"],
                    "seconds": seconds[method],
                }
                for key in scores:
                    row[key] = scores[key]
                rows.append(row)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(args.models) == len(MODELS):
        metrics_name = "metrics.csv"
    else:
        metrics_name = "metrics_" + "+".join(args.models) + ".csv"
    metrics_path = output_dir / metrics_name
    f = open(metrics_path, "w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    f.close()

    print("\nĐã lưu bảng chỉ số:", metrics_path)
    print("\nKết quả:")
    for row in rows:
        spice_text = "-"
        if row["SPICE"] is not None:
            spice_text = format(row["SPICE"], ".4f")
        print(
            format(row["model"], "22s") + " " + format(row["method"], "13s") + " " + format(row["group"], "8s")
            + " BLEU-4=" + format(row["BLEU-4"], ".4f")
            + " METEOR=" + format(row["METEOR"], ".4f")
            + " ROUGE-L=" + format(row["ROUGE-L"], ".4f")
            + " CIDEr=" + format(row["CIDEr"], ".4f")
            + " SPICE=" + spice_text
        )
    return rows, metrics_path


def build_parser():
    parser = argparse.ArgumentParser(description="Đánh giá các mô hình Flickr30k bằng nhiều thuật toán giải mã.")
    parser.add_argument("--csv_path", default=str(DATA_PATH / "results.csv"))
    parser.add_argument("--splits", default="configs/splits.json")
    parser.add_argument("--features_root", default=str(WORK_PATH / "features"))
    parser.add_argument("--checkpoint_root", default=str(WORK_PATH / "checkpoints"))
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint_override, default=[],
                        help="MODEL=PATH, có thể lặp lại; dùng khi checkpoint không nằm ở <checkpoint_root>/<MODEL>/best.pt")
    parser.add_argument("--output_dir", default=str(WORK_PATH / "evaluation"))
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS.keys()), default=list(MODELS.keys()),
                        help="Các mô hình cần chấm (mặc định: cả bốn).")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS,
                        help="Các thuật toán giải mã cần chấm (mặc định: cả bốn).")
    parser.add_argument("--sample_size", type=int, default=100,
                        help="Số ảnh chọn từ test split; 0 dùng toàn bộ test set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--diverse_beam_size", type=int, default=6)
    parser.add_argument("--diverse_groups", type=int, default=3)
    parser.add_argument("--diversity_strength", type=float, default=0.5)
    parser.add_argument("--contrastive_top_k", type=int, default=4)
    parser.add_argument("--contrastive_alpha", type=float, default=0.6)
    parser.add_argument("--max_len", type=int, default=40)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--skip_spice", action="store_true", help="Bỏ qua SPICE (cần mạng để tải CoreNLP lần đầu).")
    return parser


def main():
    args = build_parser().parse_args()
    if args.beam_size < 1:
        raise ValueError("beam_size phải >= 1")
    if args.diverse_groups < 1 or args.diverse_beam_size < args.diverse_groups:
        raise ValueError("diverse_beam_size phải >= diverse_groups >= 1")
    if args.diverse_beam_size % args.diverse_groups != 0:
        raise ValueError("diverse_beam_size phải chia hết cho diverse_groups")
    evaluate(args)


if __name__ == "__main__":
    main()