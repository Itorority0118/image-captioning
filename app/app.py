import argparse
import importlib
import os
import time

import torch
import gradio as gr

from src.data.text_preprocess import GPT2_VOCAB_SIZE, load_gpt2_tokenizer
from src.decoding.greedy import greedy_decode
from src.decoding.beam import beam_search
from src.decoding.diverse_beam import diverse_beam_search
from src.models.decoder_gpt2 import GPT2Decoder
from src.models.encoders import EXPECTED_SHAPES, get_features, get_preprocess_config, load_encoder, preprocess_image
from src.paths import checkpoint_path

MODELS = {
    "mobilenetv3_lstm": ("mobilenetv3", "lstm"),
    "efficientnet_b0_gpt2": ("efficientnet_b0", "gpt2"),
    "deit_tiny_gpt2": ("deit_tiny", "gpt2"),
    "tinyclip_gpt2": ("tinyclip", "gpt2"),
}

METHODS = ["Greedy Search", "Beam Search", "Contrastive Search", "Diverse Beam Search"]

DRIVE_CHECKPOINTS = "/content/drive/MyDrive/captioning/checkpoints"

CHECKPOINT_OVERRIDES = {}

ENCODER_CACHE = {}
PREPROCESS_CACHE = {}
DECODER_CACHE = {}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def find_checkpoint(model_name):
    if model_name in CHECKPOINT_OVERRIDES:
        return CHECKPOINT_OVERRIDES[model_name]
    candidates = [
        os.path.join(str(checkpoint_path), model_name, "best.pt"),
        os.path.join(DRIVE_CHECKPOINTS, model_name, "best.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def available_models():
    names = []
    for name in MODELS:
        if find_checkpoint(name) is not None:
            names.append(name)
    return names


def load_contrastive_function():
    for module_name in ["con_search", "contrastive"]:
        try:
            module = importlib.import_module("src.decoding." + module_name)
            return module.contrastive_search
        except ModuleNotFoundError:
            pass
    return None


def get_encoder(encoder_name, device):
    if encoder_name not in ENCODER_CACHE:
        encoder = load_encoder(encoder_name, device)
        ENCODER_CACHE[encoder_name] = encoder
        PREPROCESS_CACHE[encoder_name] = get_preprocess_config(encoder_name, encoder)
    return ENCODER_CACHE[encoder_name], PREPROCESS_CACHE[encoder_name]


def load_decoder(model_name, device):
    if model_name in DECODER_CACHE:
        return DECODER_CACHE[model_name]

    path = find_checkpoint(model_name)
    if path is None:
        raise FileNotFoundError("Chưa có checkpoint (best.pt) cho mô hình " + model_name)

    encoder_name, decoder_name = MODELS[model_name]
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("encoder") is not None and checkpoint.get("encoder") != encoder_name:
        raise ValueError("Checkpoint ghi encoder=" + str(checkpoint.get("encoder")) + ", cần " + encoder_name)
    if checkpoint.get("decoder") is not None and checkpoint.get("decoder") != decoder_name:
        raise ValueError("Checkpoint ghi decoder=" + str(checkpoint.get("decoder")) + ", cần " + decoder_name)

    feature_dim = EXPECTED_SHAPES[encoder_name][1]
    if decoder_name == "lstm":
        from src.models.decoder_lstm import LSTMDecoder

        model = LSTMDecoder(feature_dim, GPT2_VOCAB_SIZE)
    else:
        model = GPT2Decoder(feature_dim)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    info = {"path": path, "epoch": checkpoint.get("epoch"), "val_loss": checkpoint.get("val_loss")}
    DECODER_CACHE[model_name] = (model, info)
    return DECODER_CACHE[model_name]


def image_to_feature(image, encoder_name, device):
    encoder, preprocess = get_encoder(encoder_name, device)
    mean = preprocess[0]
    std = preprocess[1]
    tensor = preprocess_image(image, mean, std)
    batch = tensor.unsqueeze(0).to(device)
    features = get_features(encoder_name, encoder, batch)
    return features[0].float().cpu()


def generate_caption(image, model_name, method, max_len, beam_size, diverse_beam_size,
                     num_groups, diversity_strength, top_k, alpha):
    if image is None:
        return "Hãy tải một ảnh lên.", ""

    max_len = int(max_len)
    beam_size = int(beam_size)
    diverse_beam_size = int(diverse_beam_size)
    num_groups = int(num_groups)
    top_k = int(top_k)

    if method == "Diverse Beam Search" and diverse_beam_size % num_groups != 0:
        return "Với Diverse Beam, diverse_beam_size (" + str(diverse_beam_size) + ") phải chia hết cho num_groups (" + str(num_groups) + ").", ""

    try:
        device = get_device()
        encoder_name = MODELS[model_name][0]
        model, info = load_decoder(model_name, device)

        start = time.time()
        feature = image_to_feature(image, encoder_name, device)
        feature_seconds = time.time() - start

        start = time.time()
        tokenizer = load_gpt2_tokenizer()
        if method == "Greedy Search":
            text = greedy_decode(model, feature, max_len=max_len)
        elif method == "Beam Search":
            text = beam_search(model, feature, tokenizer=tokenizer, beam_size=beam_size, max_len=max_len)
        elif method == "Contrastive Search":
            contrastive_search = load_contrastive_function()
            if contrastive_search is None:
                return "Chưa có file Contrastive Search (con_search.py hoặc contrastive.py) trong src/decoding/.", ""
            text = contrastive_search(model, feature, max_len=max_len, top_k=top_k, alpha=float(alpha))
        else:
            groups = diverse_beam_search(
                model,
                feature,
                tokenizer=tokenizer,
                beam_size=diverse_beam_size,
                num_groups=num_groups,
                diversity_strength=float(diversity_strength),
                max_len=max_len,
            )
            lines = []
            for i in range(len(groups)):
                lines.append("Nhóm " + str(i + 1) + ": " + groups[i])
            text = "\n".join(lines)
        decode_seconds = time.time() - start

        details = "Mô hình: " + model_name + " (epoch " + str(info["epoch"]) + ", val loss " + str(info["val_loss"]) + ")"
        details = details + " | Thuật toán: " + method + " | Thiết bị: " + device
        details = details + " | Trích feature: " + str(round(feature_seconds, 2)) + " giây"
        details = details + " | Giải mã: " + str(round(decode_seconds, 2)) + " giây"
        return text, details
    except Exception as e:
        return "Lỗi: " + str(e), ""


def build_demo():
    ready = available_models()
    if len(ready) > 0:
        note = "Mô hình đã có checkpoint: " + ", ".join(ready)
        default_model = ready[0]
    else:
        note = "Chưa tìm thấy checkpoint nào. Cần có best.pt trong " + str(checkpoint_path) + "/<tên mô hình>/ (hoặc dùng --checkpoint MODEL=PATH)."
        default_model = list(MODELS.keys())[0]

    with gr.Blocks(title="Sinh chú thích ảnh tự động") as demo:
        gr.Markdown("# Sinh chú thích ảnh tự động\nTải ảnh lên, chọn mô hình và thuật toán giải mã, rồi bấm nút. Lần bấm đầu tiên chậm hơn vì phải nạp mô hình.")
        gr.Markdown(note)

        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Ảnh đầu vào")
                model_name = gr.Dropdown(choices=list(MODELS.keys()), value=default_model, label="Mô hình")
                method = gr.Dropdown(choices=METHODS, value="Greedy Search", label="Thuật toán giải mã")
                with gr.Accordion("Tham số của thuật toán", open=False):
                    max_len = gr.Slider(10, 40, value=40, step=1, label="max_len (số token tối đa, mọi thuật toán)")
                    beam_size = gr.Slider(1, 10, value=5, step=1, label="beam_size (Beam Search)")
                    diverse_beam_size = gr.Slider(2, 12, value=6, step=1, label="diverse_beam_size (Diverse Beam, phải chia hết cho num_groups)")
                    num_groups = gr.Slider(1, 6, value=3, step=1, label="num_groups (Diverse Beam)")
                    diversity_strength = gr.Slider(0.0, 2.0, value=0.5, step=0.1, label="diversity_strength (Diverse Beam)")
                    top_k = gr.Slider(1, 10, value=4, step=1, label="top_k (Contrastive)")
                    alpha = gr.Slider(0.0, 1.0, value=0.6, step=0.1, label="alpha (Contrastive, 0 là giống Greedy)")
                button = gr.Button("Sinh chú thích")
            with gr.Column():
                caption = gr.Textbox(label="Chú thích", lines=5)
                details = gr.Textbox(label="Thông tin")

        button.click(
            fn=generate_caption,
            inputs=[image, model_name, method, max_len, beam_size, diverse_beam_size, num_groups, diversity_strength, top_k, alpha],
            outputs=[caption, details],
        )
    return demo


def parse_checkpoint_override(value):
    parts = value.split("=", 1)
    if len(parts) != 2 or parts[0] not in MODELS or parts[1] == "":
        raise argparse.ArgumentTypeError("--checkpoint phải có dạng MODEL=PATH, MODEL là một trong: " + ", ".join(MODELS.keys()))
    return parts[0], parts[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Tạo đường link công khai (dùng trên Colab)")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint_override, default=[],
                        help="MODEL=PATH, có thể lặp lại")
    args = parser.parse_args()
    for model_name, path in args.checkpoint:
        CHECKPOINT_OVERRIDES[model_name] = path

    demo = build_demo()
    demo.launch(share=args.share, server_port=args.port)