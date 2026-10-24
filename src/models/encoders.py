import os
import sys

import numpy as np
import torch
import timm
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor

ENCODER_NAMES = ["mobilenetv3", "efficientnet_b0", "deit_tiny", "tinyclip"]

TIMM_NAMES = {
    "mobilenetv3": "mobilenetv3_small_100",
    "efficientnet_b0": "efficientnet_b0",
    "deit_tiny": "deit_tiny_patch16_224",
}

TINYCLIP_NAME = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"

EXPECTED_SHAPES = {
    "mobilenetv3": (49, 576),
    "efficientnet_b0": (49, 1280),
    "deit_tiny": (197, 192),
    "tinyclip": (50, 512),
}

IMAGE_SIZE = 224

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

def count_params(model):
    total = 0
    for p in model.parameters():
        total = total + p.numel()
    return total

def load_encoder(name, device):
    if name in TIMM_NAMES:
        model = timm.create_model(TIMM_NAMES[name], pretrained=True)
    elif name == "tinyclip":
        model = CLIPModel.from_pretrained(TINYCLIP_NAME)
    else:
        raise ValueError("Encoder không hợp lệ: " + str(name) + ". Chọn một trong: " + str(ENCODER_NAMES))
    model.eval()
    model.to(device)
    return model

def get_preprocess_config(name, model):
    if name == "tinyclip":
        try:
            processor = CLIPImageProcessor.from_pretrained(TINYCLIP_NAME)
            mean = list(processor.image_mean)
            std = list(processor.image_std)
        except Exception as e:
            print("Không đọc được cấu hình CLIP, dùng giá trị CLIP chuẩn. Lỗi:", e)
            mean = CLIP_MEAN
            std = CLIP_STD
    else:
        cfg = model.pretrained_cfg
        mean = list(cfg["mean"])
        std = list(cfg["std"])
    return mean, std

def preprocess_image(img, mean, std):
    img = img.convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    arr = (arr - mean_arr) / std_arr

    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)

def get_features(name, model, batch):
    with torch.no_grad():
        if name == "tinyclip":
            out = model.vision_model(pixel_values=batch)
            features = out.last_hidden_state
        elif name == "deit_tiny":
            features = model.forward_features(batch)
        else:
            maps = model.forward_features(batch)
            features = maps.flatten(2).transpose(1, 2)
    return features

if __name__ == "__main__":
    image_dir = os.path.join("data", "flickr30k_images")
    if len(sys.argv) >= 2:
        image_dir = sys.argv[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Thiết bị:", device)

    files = []
    for f in sorted(os.listdir(image_dir)):
        if f.lower().endswith(".jpg"):
            files.append(f)
        if len(files) == 2:
            break
    print("Ảnh thử:", files)

    for name in ENCODER_NAMES:
        print("=" * 60)
        print("Encoder:", name)
        model = load_encoder(name, device)
        mean, std = get_preprocess_config(name, model)
        print("mean:", mean)
        print("std:", std)

        if name == "tinyclip":
            vc = model.config.vision_config
            print("Cấu hình ViT: số lớp", vc.num_hidden_layers, "| chiều ẩn", vc.hidden_size, "| patch", vc.patch_size, "| cỡ ảnh", vc.image_size)
            print("Số tham số encoder ảnh (triệu):", round(count_params(model.vision_model) / 1000000, 1), "(bài báo: 40M)")
            try:
                processor = CLIPImageProcessor.from_pretrained(TINYCLIP_NAME)
                print("Bộ xử lý ảnh CLIP: size", processor.size, "| crop_size", processor.crop_size)
            except Exception as e:
                print("Không đọc được bộ xử lý ảnh CLIP:", e)
        else:
            cfg = model.pretrained_cfg
            print("Trọng số:", cfg.get("hf_hub_id"))
            print("input_size:", cfg.get("input_size"), "| interpolation:", cfg.get("interpolation"), "| crop_pct:", cfg.get("crop_pct"))
            print("Số tham số (triệu):", round(count_params(model) / 1000000, 1), "(bài báo: V3-Small 2.5M, B0 5.3M, DeiT-Ti 5M)")

        tensors = []
        for f in files:
            img = Image.open(os.path.join(image_dir, f))
            tensors.append(preprocess_image(img, mean, std))
        batch = torch.stack(tensors).to(device)

        features = get_features(name, model, batch)
        one_shape = tuple(features.shape[1:])
        if one_shape == EXPECTED_SHAPES[name]:
            result = "OK"
        else:
            result = "LỖI, mong đợi " + str(EXPECTED_SHAPES[name])
        print("Shape một ảnh:", one_shape, "|", result)