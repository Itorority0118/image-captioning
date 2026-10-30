import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

from src.paths import DATA_PATH, get_feature_path, get_checkpoint_path
from src.models.encoders import EXPECTED_SHAPES
from src.models.decoder_gpt2 import GPT2Decoder
from src.data.dataset import Flickr30kDataset
from src.data.text_preprocess import PAD_TOKEN, decode_caption, save_vocab


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_decoder(decoder_name, feature_dim, vocab_size):
    if decoder_name == "gpt2":
        return GPT2Decoder(feature_dim=feature_dim)
    if decoder_name == "lstm":
        from src.models.decoder_lstm import LSTMDecoder
        return LSTMDecoder(feature_dim=feature_dim, vocab_size=vocab_size)
    raise ValueError(f"Decoder không hợp lệ: {decoder_name}")


def compute_loss(model, features, captions, criterion, pad_id):
    inp = captions[:, :-1]
    targets = captions[:, 1:].contiguous()
    mask = (inp != pad_id).long()

    logits = model(features, inp, mask)
    if isinstance(logits, tuple):
        logits = logits[0]

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

    encoder_name = args.encoder or cfg.get("encoder", "mobilenetv3")
    decoder_name = args.decoder or cfg.get("decoder", "gpt2")
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
    feat_dir = get_feature_path(encoder_name)
    ckpt_dir = get_checkpoint_path(encoder_name, decoder_name)

    # Khởi tạo Dataset trực tiếp từ src/data/dataset.py
    train_ds = Flickr30kDataset(csv_path, splits_path, feat_dir, split="train", max_len=max_len)
    vocab = train_ds.vocab
    val_ds = Flickr30kDataset(csv_path, splits_path, feat_dir, split="val", vocab=vocab, max_len=max_len)

    save_vocab(vocab, ckpt_dir / "vocab.json")
    pad_id = vocab[PAD_TOKEN]

    if args.debug:
        train_ds = Subset(train_ds, range(min(32, len(train_ds))))
        val_ds = Subset(val_ds, range(min(32, len(val_ds))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Thiết bị: {device} | Mô hình: {encoder_name}_{decoder_name} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    _, feature_dim = EXPECTED_SHAPES[encoder_name]
    model = build_decoder(decoder_name, feature_dim, len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, start_t = 0.0, time.time()

        for step, (features, captions) in enumerate(train_loader, 1):
            features, captions = features.to(device), captions.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model, features, captions, criterion, pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

            if step % 100 == 0 or (args.debug and step == len(train_loader)):
                print(f"  [Epoch {epoch}/{epochs}] Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, captions in val_loader:
                features, captions = features.to(device), captions.to(device)
                val_loss += compute_loss(model, features, captions, criterion, pad_id).item()

        avg_train = train_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))
        print(f"=> Epoch {epoch}/{epochs} ({time.time() - start_t:.1f}s) | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        ckpt = {
            "epoch": epoch,
            "encoder": encoder_name,
            "decoder": decoder_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val,
            "config": cfg,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if avg_val <= best_val_loss:
            best_val_loss = avg_val
            torch.save(ckpt, ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()