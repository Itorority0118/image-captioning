import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from src.models.encoders import ENCODER_NAMES, EXPECTED_SHAPES
from src.models.encoders import load_encoder, get_preprocess_config, preprocess_image, get_features
from src.paths import feature_path

def feature_file_path(out_dir, image_name):
    base = os.path.splitext(image_name)[0]
    return os.path.join(out_dir, base + ".npy")

def save_feature(array, out_dir, image_name):
    final_path = feature_file_path(out_dir, image_name)
    tmp_path = final_path[:-4] + ".tmp.npy"
    np.save(tmp_path, array)
    os.replace(tmp_path, final_path)


def extract_feature(img, name, model, device):
    mean, std = get_preprocess_config(name, model)
    tensor = preprocess_image(img, mean, std)
    batch = tensor.unsqueeze(0).to(device)
    features = get_features(name, model, batch)
    return features[0].cpu().numpy().astype(np.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=ENCODER_NAMES)
    parser.add_argument("--image_dir", default=os.path.join("data", "flickr30k_images"))
    parser.add_argument("--splits", default=os.path.join("configs", "splits.json"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="chỉ làm N ảnh đầu để thử (0 là làm hết)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Thiết bị:", device)
    print("Encoder:", args.encoder)

    f = open(args.splits, encoding="utf-8")
    splits = json.load(f)
    f.close()
    names = splits["train"] + splits["val"] + splits["test"]
    if args.limit > 0:
        names = names[:args.limit]
    print("Số ảnh trong danh sách:", len(names))

    out_dir = os.path.join(str(feature_path), args.encoder)
    os.makedirs(out_dir, exist_ok=True)
    print("Thư mục lưu feature:", out_dir)

    todo = []
    already = 0
    missing = 0
    for name in names:
        if os.path.exists(feature_file_path(out_dir, name)):
            already = already + 1
        elif not os.path.exists(os.path.join(args.image_dir, name)):
            missing = missing + 1
        else:
            todo.append(name)
    print("Đã có sẵn:", already, "| thiếu file ảnh:", missing, "| cần làm:", len(todo))

    if len(todo) > 0:
        model = load_encoder(args.encoder, device)
        mean, std = get_preprocess_config(args.encoder, model)

    failed = []
    biggest = 0.0
    done = 0
    start_time = time.time()
    batch_count = 0

    for start in range(0, len(todo), args.batch_size):
        batch_names = todo[start:start + args.batch_size]
        tensors = []
        ok_names = []
        for name in batch_names:
            try:
                img = Image.open(os.path.join(args.image_dir, name))
                tensors.append(preprocess_image(img, mean, std))
                img.close()
                ok_names.append(name)
            except Exception as e:
                failed.append(name)
        if len(tensors) == 0:
            continue

        batch = torch.stack(tensors).to(device)
        features = get_features(args.encoder, model, batch)
        arr = features.cpu().numpy()
        biggest = max(biggest, float(np.abs(arr).max()))
        arr = arr.astype(np.float16)

        for i in range(len(ok_names)):
            save_feature(arr[i], out_dir, ok_names[i])
        done = done + len(ok_names)

        batch_count = batch_count + 1
        if batch_count % 20 == 0:
            used = round(time.time() - start_time)
            print("Đã làm", done, "/", len(todo), "ảnh, đã chạy", used, "giây", flush=True)

    used = round(time.time() - start_time)
    print("=" * 50)
    print("Xong. Làm mới:", done, "ảnh trong", used, "giây | ảnh lỗi:", len(failed))
    if len(failed) > 0:
        print("Danh sách ảnh lỗi (5 ảnh đầu):", failed[:5])
    if biggest > 0:
        print("Giá trị tuyệt đối lớn nhất của feature:", round(biggest, 2))
        if biggest > 60000:
            print("CẢNH BÁO: gần chạm giới hạn của float16")

    if len(names) > 0:
        check_path = None
        for name in names:
            if os.path.exists(feature_file_path(out_dir, name)):
                check_path = feature_file_path(out_dir, name)
                break
        if check_path is not None:
            arr = np.load(check_path)
            if tuple(arr.shape) == EXPECTED_SHAPES[args.encoder]:
                result = "OK"
            else:
                result = "LỖI, mong đợi " + str(EXPECTED_SHAPES[args.encoder])
            print("Kiểm tra file", os.path.basename(check_path), "| shape:", arr.shape, "| kiểu:", arr.dtype, "|", result)

if __name__ == "__main__":
    main()