import torch
from src.data.text_preprocess import SPECIAL_TOKEN_ID, decode_caption


@torch.no_grad()
def greedy_decode(model, features, max_len=40):
    model.eval()
    device = next(model.parameters()).device

    if features.dim() == 2:
        features = features.unsqueeze(0)
    features = features.to(device, dtype=torch.float32)

    input_ids = torch.tensor([[SPECIAL_TOKEN_ID]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        logits = model(features, input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]

        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if next_id.item() == SPECIAL_TOKEN_ID:
            break

        input_ids = torch.cat([input_ids, next_id], dim=1)

    return decode_caption(input_ids[0])