import torch


def next_token_logits(model, features, input_ids):
    if hasattr(model, "next_token_logits"):
        return model.next_token_logits(features, input_ids)

    logits = model(features, input_ids)
    if isinstance(logits, tuple):
        logits = logits[0]
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or logits.shape[:2] != input_ids.shape
        or logits.shape[-1] == 0
    ):
        raise ValueError("Decoder must return logits shaped (batch, sequence_length, vocab_size)")
    return logits[:, -1, :]
