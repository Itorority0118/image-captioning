"""Sinh caption bằng Beam Search với decoder và tokenizer GPT-2."""

import math

import torch

from src.data.text_preprocess import decode_caption, load_gpt2_tokenizer


def _beam_score(beam, length_penalty):
    token_ids, log_probability, _ = beam
    length = len(token_ids) - 1  # Không tính BOS; tính EOS nếu đã sinh.
    penalty = ((5.0 + length) / 6.0) ** length_penalty
    return log_probability / penalty


def _beam_search_one(model, features, tokenizer, beam_size, max_len, length_penalty):
    # Mỗi beam gồm: (danh sách token, tổng log xác suất, đã gặp EOS).
    # BOS và EOS của GPT-2 có thể cùng ID, nên cần lưu trạng thái riêng.
    beams = [([tokenizer.bos_token_id], 0.0, False)]

    for _ in range(max_len - 1):
        active_beams = [beam for beam in beams if not beam[2]]
        if not active_beams:
            break

        # Các beam đang mở luôn cùng độ dài: gộp thành một batch để gọi model.
        input_ids = torch.tensor(
            [beam[0] for beam in active_beams],
            dtype=torch.long,
            device=features.device,
        )
        beam_features = features.expand(len(active_beams), -1, -1)
        logits = model(beam_features, input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or logits.shape[:2] != input_ids.shape
            or logits.shape[-1] == 0
        ):
            raise ValueError(
                "Decoder phải trả logits có shape (batch, sequence_length, vocab_size)"
            )

        # Cộng log xác suất để tránh nhân nhiều xác suất rất nhỏ.
        log_probs = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
        previous_scores = log_probs.new_tensor([beam[1] for beam in active_beams])
        candidate_scores = previous_scores.unsqueeze(1) + log_probs
        top_scores, top_indices = torch.topk(
            candidate_scores.flatten(),
            k=min(beam_size, candidate_scores.numel()),
        )

        # Giữ nguyên câu đã kết thúc, không sinh thêm token sau EOS.
        candidates = [beam for beam in beams if beam[2]]
        vocab_size = log_probs.shape[-1]
        for score, index in zip(top_scores.tolist(), top_indices.tolist()):
            if not math.isfinite(score):
                continue
            parent_index, token_id = divmod(index, vocab_size)
            token_ids = active_beams[parent_index][0] + [token_id]
            candidates.append((token_ids, score, token_id == tokenizer.eos_token_id))

        if not candidates:
            raise ValueError("Decoder không cung cấp token tiếp theo có xác suất hợp lệ")

        # Mọi câu mở rộng có cùng độ dài, nên top-k phía trên không làm mất
        # ứng viên tốt hơn khi so sánh với các câu đã kết thúc.
        beams = sorted(
            candidates,
            key=lambda beam: _beam_score(beam, length_penalty),
            reverse=True,
        )[:beam_size]

    return decode_caption(beams[0][0], tokenizer=tokenizer)


@torch.inference_mode()
def beam_search(
    model,
    features,
    tokenizer=None,
    beam_size=5,
    max_len=40,
    length_penalty=1.0,
):
    """Sinh một hoặc nhiều caption bằng Beam Search.

    Args:
        model: Decoder nhận ``model(features, input_ids)`` và trả logits.
        features: Tensor ``(regions, feature_dim)`` cho một ảnh hoặc
            ``(batch, regions, feature_dim)`` cho nhiều ảnh.
        tokenizer: Tokenizer GPT-2; tự nạp tokenizer của dự án nếu bỏ qua.
        beam_size: Số câu ứng viên tối đa được giữ sau mỗi bước.
        max_len: Số token tối đa, tính cả BOS và EOS nếu có.
        length_penalty: Số mũ của hệ số ``((5 + length) / 6)`` khi chia điểm
            log xác suất; đặt 0 để chỉ so sánh tổng log xác suất.

    Returns:
        Chuỗi caption nếu features có 2 chiều; danh sách caption nếu có 3 chiều.
        Câu chưa gặp EOS sẽ dừng khi đạt max_len.
    """
    if not isinstance(beam_size, int) or isinstance(beam_size, bool) or beam_size < 1:
        raise ValueError("beam_size phải là số nguyên >= 1")
    if not isinstance(max_len, int) or isinstance(max_len, bool) or max_len < 2:
        raise ValueError("max_len phải là số nguyên >= 2")
    if not math.isfinite(length_penalty) or length_penalty < 0:
        raise ValueError("length_penalty phải là số hữu hạn >= 0")
    if not isinstance(features, torch.Tensor):
        raise TypeError("features phải là torch.Tensor")
    if features.ndim not in (2, 3) or min(features.shape[-2:]) == 0:
        raise ValueError(
            "features phải có shape (regions, feature_dim) hoặc "
            "(batch, regions, feature_dim), với regions và feature_dim > 0"
        )

    if tokenizer is None:
        tokenizer = load_gpt2_tokenizer()
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("tokenizer cần có bos_token_id và eos_token_id")

    single_image = features.ndim == 2
    feature_batch = features.unsqueeze(0) if single_image else features
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else features.device
    dtype = (
        parameter.dtype
        if parameter is not None and parameter.is_floating_point()
        else torch.float32
    )
    feature_batch = feature_batch.to(device=device, dtype=dtype)

    was_training = model.training
    try:
        model.eval()
        captions = [
            _beam_search_one(
                model,
                image_features.unsqueeze(0),
                tokenizer,
                beam_size,
                max_len,
                length_penalty,
            )
            for image_features in feature_batch
        ]
    finally:
        model.train(was_training)

    return captions[0] if single_image else captions


# Tên tương ứng với greedy_decode trong cùng thư mục.
beam_decode = beam_search
