import argparse, os, random, shutil, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml

from src.paths import DATA_PATH, get_feature_path, get_checkpoint_path
from src.models.encoders import EXPECTED_SHAPES
from src.models.decoder_gpt2 import GPT2Decoder
from src.data.dataset import Flickr30kDataset
from src.data.text_preprocess import GPT2_VOCAB_SIZE


def forward_loss(model, batch, device):
    feat = F.dropout(batch["image_features"].to(device, non_blocking=True), 0.2, model.training)
    inp = batch["decoder_input_ids"].to(device, non_blocking=True)
    mask = batch["attention_mask"].to(device, non_blocking=True)
    targets = batch["target_ids"].to(device, non_blocking=True).masked_fill(
        batch["loss_mask"].to(device, non_blocking=True) == 0, -100
    )
    with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
        logits = model(feat, inp, mask) if hasattr(model, "gpt2") else model(feat, inp)
        logits = logits[0] if isinstance(logits, tuple) else logits
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=-100, label_smoothing=(0.1 if model.training else 0.0)
        )


def main():
    p = argparse.ArgumentParser()
    for k, d, t in [("--config", "configs/base.yaml", str), ("--encoder", None, str), ("--decoder", None, str),
                    ("--splits", None, str), ("--csv_path", None, str), ("--epochs", None, int),
                    ("--batch_size", None, int), ("--lr", None, float)]:
        p.add_argument(k, default=d, type=t)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    enc = args.encoder or cfg.get("encoder", "efficientnet_b0")
    dec = args.decoder or cfg.get("decoder", "gpt2")
    max_len = int(cfg.get("max_len", 40))
    bs = 8 if args.debug else (args.batch_size or int(cfg.get("batch_size", 64)))
    lr = args.lr or float(cfg.get("learning_rate", 3e-4))
    epochs = 2 if args.debug else (args.epochs or int(cfg.get("epochs", 10)))

    seed = int(cfg.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    splits = args.splits or ("configs/splits_sample.json" if args.debug and os.path.exists("configs/splits_sample.json") else "configs/splits.json")
    csv_path = args.csv_path or str(DATA_PATH / "results.csv")
    feat_dir, ckpt_dir, shape = get_feature_path(enc), get_checkpoint_path(enc, dec), EXPECTED_SHAPES[enc]

    train_ds = Flickr30kDataset(csv_path, splits, feat_dir, "train", max_len, shape)
    val_ds = Flickr30kDataset(csv_path, splits, feat_dir, "val", max_len, shape)
    if args.debug:
        train_ds, val_ds = Subset(train_ds, range(min(32, len(train_ds)))), Subset(val_ds, range(min(32, len(val_ds))))

    kw = {"num_workers": 2, "pin_memory": True} if device.type == "cuda" and not args.debug else {}
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False, **kw)

    if dec == "lstm":
        from src.models.decoder_lstm import LSTMDecoder
        model = LSTMDecoder(shape[1], GPT2_VOCAB_SIZE).to(device)
    else:
        model = GPT2Decoder(shape[1]).to(device)
        for n, param in model.gpt2.transformer.named_parameters():
            param.requires_grad = any(k in n for k in ("h.10.", "h.11.", "ln_f"))

    gpt2_p = [p for n, p in model.named_parameters() if p.requires_grad and "gpt2." in n]
    proj_p = [p for n, p in model.named_parameters() if p.requires_grad and "gpt2." not in n]
    params = proj_p + gpt2_p
    print(f"Thiết bị: {device} | Mô hình: {enc}_{dec} | Train: {len(train_ds)} | Tham số học: {sum(p.numel() for p in params):,}")

    opt = torch.optim.AdamW([{"params": proj_p, "lr": lr}, {"params": gpt2_p, "lr": lr * 0.1}], weight_decay=0.05)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    best_val, start_ep = float("inf"), 1
    drive_dir = next((Path(p) / "captioning/checkpoints" / f"{enc}_{dec}" for p in ("/content/drive/MyDrive", "/content/gdrive/MyDrive") if os.path.exists(p)), None)
    best_ckpt = ckpt_dir / "best.pt"

    if not args.debug and not best_ckpt.exists() and drive_dir and (drive_dir / "best.pt").exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_dir / "best.pt", best_ckpt)

    if not args.debug and best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        if ckpt.get("encoder") == enc and ckpt.get("decoder") == dec:
            model.load_state_dict(ckpt["model_state_dict"])
            start_ep, best_val = ckpt["epoch"] + 1, ckpt.get("best_val", float("inf"))
            print(f"Khôi phục {best_ckpt} (Epoch {ckpt['epoch']}, Best Val: {best_val:.4f})")

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - start_ep + 1) * len(train_loader), eta_min=1e-6)

    for ep in range(start_ep, epochs + 1):
        model.train()
        t_loss, t0 = 0.0, time.time()
        for step, batch in enumerate(train_loader, 1):
            opt.zero_grad(set_to_none=True)
            loss = forward_loss(model, batch, device)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            t_loss += loss.item()
            if step % 200 == 0 or (args.debug and step == len(train_loader)):
                print(f"  [Epoch {ep}/{epochs}] Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)

        model.eval()
        with torch.no_grad():
            v_loss = sum(forward_loss(model, b, device).item() for b in val_loader)

        avg_t, avg_v = t_loss / max(1, len(train_loader)), v_loss / max(1, len(val_loader))
        print(f"=> Epoch {ep}/{epochs} ({time.time() - t0:.1f}s) | Train: {avg_t:.4f} | Val: {avg_v:.4f}")

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {"epoch": ep, "encoder": enc, "decoder": dec, "model_state_dict": model.state_dict(), "val_loss": avg_v, "best_val": min(best_val, avg_v)}
        torch.save(state, ckpt_dir / "last.pt")

        if avg_v < best_val:
            best_val = avg_v
            torch.save(state, best_ckpt)
            if drive_dir and not args.debug:
                drive_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(best_ckpt, drive_dir / "best.pt")
                print(f"  [Drive] Đã cập nhật best.pt (Val Loss: {best_val:.4f})")


if __name__ == "__main__":
    main()