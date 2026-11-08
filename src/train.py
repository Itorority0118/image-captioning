import argparse
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

from src.paths import DATA_PATH, get_feature_path, get_checkpoint_path
from src.models.encoders import EXPECTED_SHAPES
from src.models.decoder_gpt2 import GPT2Decoder
from src.data.dataset import Flickr30kDataset
from src.data.text_preprocess import GPT2_VOCAB_SIZE


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_decoder(decoder_name, feature_dim):
    if decoder_name == "gpt2":
        return GPT2Decoder(feature_dim=feature_dim)
    if decoder_name == "lstm":
        from src.models.decoder_lstm import LSTMDecoder
        return LSTMDecoder(feature_dim=feature_dim, vocab_size=GPT2_VOCAB_SIZE)
    raise ValueError(f"Decoder không hợp lệ: {decoder_name}")


def forward_loss(model, batch, criterion, device):
    features = batch["image_features"].to(device)
    inp = batch["decoder_input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    targets = batch["target_ids"].to(device)
    loss_mask = batch["loss_mask"].to(device)

    logits = model(features, inp, mask) if getattr(model, "gpt2", None) else model(features, inp)
    if isinstance(logits, tuple):
        logits = logits[0]

    targets = targets.masked_fill(loss_mask == 0, -100)
    return criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--encoder", choices=["mobilenetv3", "efficientnet_b0", "deit_tiny", "tinyclip"])
    parser.add_argument("--decoder", choices=["lstm", "gpt2"])
    parser.add_argument("--splits", default=None)
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    encoder = args.encoder or cfg.get("encoder", "efficientnet_b0")
    decoder = args.decoder or cfg.get("decoder", "gpt2")
    max_len = int(cfg.get("max_len", 40))
    batch_size = args.batch_size or int(cfg.get("batch_size", 64))
    lr = args.lr or float(cfg.get("learning_rate", 1e-4))
    epochs = args.epochs or int(cfg.get("epochs", 10))

    if args.debug:
        epochs, batch_size = 2, min(batch_size, 8)

    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits_path = args.splits or (
        "configs/splits_sample.json" if args.debug and os.path.exists("configs/splits_sample.json") else "configs/splits.json"
    )
    csv_path = args.csv_path or str(DATA_PATH / "results.csv")
    feat_dir = get_feature_path(encoder)
    ckpt_dir = get_checkpoint_path(encoder, decoder)
    expected_shape = EXPECTED_SHAPES[encoder]

    train_ds = Flickr30kDataset(csv_path, splits_path, feat_dir, "train", max_len, expected_shape)
    val_ds = Flickr30kDataset(csv_path, splits_path, feat_dir, "val", max_len, expected_shape)

    if args.debug:
        train_ds = Subset(train_ds, range(min(32, len(train_ds))))
        val_ds = Subset(val_ds, range(min(32, len(val_ds))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Thiết bị: {device} | Mô hình: {encoder}_{decoder} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = build_decoder(decoder, expected_shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_val = float("inf")
    start_epoch = 1

    drive_base = next((Path(p) for p in ["/content/drive/MyDrive", "/content/gdrive/MyDrive"] if os.path.exists(p)), None)
    drive_ckpt_dir = (drive_base / "captioning" / "checkpoints" / f"{encoder}_{decoder}") if drive_base else None

    last_ckpt = ckpt_dir / "last.pt"
    if not last_ckpt.exists() and not args.debug and drive_ckpt_dir and (drive_ckpt_dir / "last.pt").exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for pt in drive_ckpt_dir.glob("*.pt"):
            shutil.copy2(pt, ckpt_dir / pt.name)

    if last_ckpt.exists() and not args.debug:
        ckpt = torch.load(last_ckpt, map_location=device)
        if ckpt.get("encoder") == encoder and ckpt.get("decoder") == decoder:
            print(f"Khôi phục tiến trình từ {last_ckpt} (Epoch {ckpt['epoch']})...")
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val = ckpt.get("best_val", ckpt.get("val_loss", float("inf")))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss, t0 = 0.0, time.time()

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            loss = forward_loss(model, batch, criterion, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

            if step % 100 == 0 or (args.debug and step == len(train_loader)):
                print(f"  [Epoch {epoch}/{epochs}] Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)

        model.eval()
        with torch.no_grad():
            val_loss = sum(forward_loss(model, b, criterion, device).item() for b in val_loader)

        avg_train = train_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))
        is_best = avg_val <= best_val
        if is_best:
            best_val = avg_val

        print(f"=> Epoch {epoch}/{epochs} ({time.time() - t0:.1f}s) | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        ckpt = {
            "epoch": epoch,
            "encoder": encoder,
            "decoder": decoder,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val,
            "best_val": best_val,
            "config": cfg,
        }
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, ckpt_dir / "last.pt")
        if is_best:
            torch.save(ckpt, ckpt_dir / "best.pt")

        if drive_ckpt_dir and not args.debug:
            drive_ckpt_dir.mkdir(parents=True, exist_ok=True)
            for name in ("last.pt", "best.pt") if is_best else ("last.pt",):
                tmp_path = drive_ckpt_dir / f"{name}.tmp"
                shutil.copy2(ckpt_dir / name, tmp_path)
                tmp_path.replace(drive_ckpt_dir / name)
            print(f"  Đã sao lưu an toàn epoch {epoch} lên Drive: {drive_ckpt_dir}")


if __name__ == "__main__":
    main()