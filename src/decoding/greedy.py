import torch
from src.data.text_preprocess import SPECIAL_TOKEN_ID, decode_caption


@torch.no_grad()
def greedy_decode(model, features, max_len=40):
    model.eval()
    device = next(model.parameters()).device
    features = (features.unsqueeze(0) if features.dim() == 2 else features).to(device, dtype=torch.float32)
    input_ids = torch.tensor([[SPECIAL_TOKEN_ID]], device=device)

    for _ in range(max_len - 1):
        logits = model(features, input_ids)
        next_id = (logits[0] if isinstance(logits, tuple) else logits)[:, -1].argmax(-1, keepdim=True)
        if next_id.item() == SPECIAL_TOKEN_ID:
            break
        input_ids = torch.cat([input_ids, next_id], dim=1)

    return decode_caption(input_ids[0])


if __name__ == "__main__":
    import sys, numpy as np
    from pathlib import Path
    from src.paths import get_feature_path, get_checkpoint_path
    from src.models.encoders import EXPECTED_SHAPES
    from src.models.decoder_gpt2 import GPT2Decoder

    enc = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].endswith(".pt") else "efficientnet_b0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    drive_ckpt = Path(f"/content/drive/MyDrive/captioning/checkpoints/{enc}_gpt2/best.pt")
    local_ckpt = get_checkpoint_path(enc, "gpt2") / "best.pt"
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].endswith(".pt") else (
        drive_ckpt if drive_ckpt.exists() else local_ckpt
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT2Decoder(EXPECTED_SHAPES[enc][1]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    work_feat = Path("/content/work/features")
    feat_dir = work_feat if work_feat.exists() and any(work_feat.glob("*.npy")) else get_feature_path(enc)
    
    feat_file = next(feat_dir.glob("*.npy"))
    print(f"[{feat_file.stem}]: {greedy_decode(model, torch.from_numpy(np.load(feat_file)))}")