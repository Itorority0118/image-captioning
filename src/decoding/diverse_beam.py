"""Sinh nhiều caption có tính đa dạng bằng Diverse Beam Search.

Diverse Beam Search chia beam thành nhiều nhóm. Từng nhóm được giải mã
tuần tự; token đã được các nhóm trước chọn ở cùng bước bị trừ điểm để
khuyến khích các nhóm tạo ra caption khác nhau.
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

import torch

from src.data.text_preprocess import decode_caption, load_gpt2_tokenizer


@dataclass
class _Beam:
    token_ids: List[int]
    log_probability: float
    search_score: float
    finished: bool
    # Token được sinh ở bước hiện tại; None với beam đã kết thúc từ trước.
    last_token: Optional[int] = None


def _normalized_score(beam, length_penalty):
    """Chuẩn hóa điểm theo độ dài, không tính BOS nhưng tính EOS."""
    length = max(1, len(beam.token_ids) - 1)
    penalty = ((5.0 + length) / 6.0) ** length_penalty
    return beam.search_score / penalty


def _diverse_search_one(
    model,
    features,
    tokenizer,
    beam_size,
    num_groups,
    diversity_strength,
    max_len,
    length_penalty,
):
    group_size = beam_size // num_groups
    groups = [
        [_Beam([tokenizer.bos_token_id], 0.0, 0.0, False)]
        for _ in range(num_groups)
    ]

    for _ in range(max_len - 1):
        if all(beam.finished for group in groups for beam in group):
            break

        # Chỉ đếm token được sinh ở bước này bởi các nhóm đứng trước.
        previous_group_tokens = Counter()
        next_groups = []

        for group in groups:
            active_beams = [beam for beam in group if not beam.finished]

            # Beam đã kết thúc được giữ nguyên, không sinh thêm token.
            candidates = [
                _Beam(
                    beam.token_ids,
                    beam.log_probability,
                    beam.search_score,
                    True,
                    None,
                )
                for beam in group
                if beam.finished
            ]

            if active_beams:
                input_ids = torch.tensor(
                    [beam.token_ids for beam in active_beams],
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
                        "Decoder phải trả logits có shape "
                        "(batch, sequence_length, vocab_size)"
                    )

                log_probs = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
                vocab_size = log_probs.shape[-1]

                # Phạt token mà các nhóm trước vừa chọn tại cùng bước.
                diversity_penalty = log_probs.new_zeros(vocab_size)
                if previous_group_tokens:
                    token_ids = torch.tensor(
                        list(previous_group_tokens.keys()),
                        dtype=torch.long,
                        device=log_probs.device,
                    )
                    token_counts = log_probs.new_tensor(
                        [previous_group_tokens[token_id] for token_id in token_ids.tolist()]
                    )
                    diversity_penalty[token_ids] = token_counts

                previous_scores = log_probs.new_tensor(
                    [beam.search_score for beam in active_beams]
                )
                candidate_scores = (
                    previous_scores.unsqueeze(1)
                    + log_probs
                    - diversity_strength * diversity_penalty.unsqueeze(0)
                )

                top_scores, top_indices = torch.topk(
                    candidate_scores.flatten(),
                    k=min(group_size, candidate_scores.numel()),
                )

                for adjusted_score, index in zip(
                    top_scores.tolist(), top_indices.tolist()
                ):
                    if not math.isfinite(adjusted_score):
                        continue

                    parent_index, token_id = divmod(index, vocab_size)
                    model_log_prob = float(log_probs[parent_index, token_id].item())
                    if not math.isfinite(model_log_prob):
                        continue

                    parent = active_beams[parent_index]
                    candidates.append(
                        _Beam(
                            token_ids=parent.token_ids + [token_id],
                            log_probability=parent.log_probability + model_log_prob,
                            search_score=adjusted_score,
                            finished=(token_id == tokenizer.eos_token_id),
                            last_token=token_id,
                        )
                    )

            if not candidates:
                raise ValueError(
                    "Decoder không cung cấp token tiếp theo có xác suất hợp lệ"
                )

            selected = sorted(
                candidates,
                key=lambda beam: _normalized_score(beam, length_penalty),
                reverse=True,
            )[:group_size]
            next_groups.append(selected)

            for beam in selected:
                if beam.last_token is not None:
                    previous_group_tokens[beam.last_token] += 1

        groups = next_groups

    # Trả caption có điểm cao nhất của mỗi nhóm để giữ tính đa dạng giữa nhóm.
    return [
        decode_caption(
            max(group, key=lambda beam: _normalized_score(beam, length_penalty)).token_ids,
            tokenizer=tokenizer,
        )
        for group in groups
    ]


@torch.inference_mode()
def diverse_beam_search(
    model,
    features,
    tokenizer=None,
    beam_size=6,
    num_groups=3,
    diversity_strength=0.5,
    max_len=40,
    length_penalty=1.0,
):
    """Sinh các caption đa dạng bằng Grouped Diverse Beam Search.

    Args:
        model: Decoder nhận ``model(features, input_ids)`` và trả logits.
        features: Tensor ``(regions, feature_dim)`` cho một ảnh hoặc
            ``(batch, regions, feature_dim)`` cho nhiều ảnh.
        tokenizer: Tokenizer GPT-2; tự nạp tokenizer của dự án nếu bỏ qua.
        beam_size: Tổng số beam, phải chia hết cho ``num_groups``.
        num_groups: Số nhóm beam. Kết quả có một caption tốt nhất mỗi nhóm.
        diversity_strength: Mức phạt cho token được nhóm trước chọn ở cùng
            bước; đặt 0 để tắt phạt đa dạng.
        max_len: Số token tối đa, tính cả BOS và EOS nếu có.
        length_penalty: Số mũ của hệ số ``((5 + length) / 6)`` khi xếp hạng.

    Returns:
        Với một ảnh, trả ``list[str]`` gồm một caption cho mỗi nhóm.
        Với batch ảnh, trả ``list[list[str]]`` theo thứ tự ảnh.
    """
    for name, value in (("beam_size", beam_size), ("num_groups", num_groups)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} phải là số nguyên >= 1")
    if num_groups > beam_size or beam_size % num_groups != 0:
        raise ValueError("beam_size phải chia hết cho num_groups và >= num_groups")
    if (
        isinstance(max_len, bool)
        or not isinstance(max_len, int)
        or max_len < 2
    ):
        raise ValueError("max_len phải là số nguyên >= 2")
    if (
        isinstance(diversity_strength, bool)
        or not isinstance(diversity_strength, (int, float))
        or not math.isfinite(diversity_strength)
        or diversity_strength < 0
    ):
        raise ValueError("diversity_strength phải là số hữu hạn >= 0")
    if (
        isinstance(length_penalty, bool)
        or not isinstance(length_penalty, (int, float))
        or not math.isfinite(length_penalty)
        or length_penalty < 0
    ):
        raise ValueError("length_penalty phải là số hữu hạn >= 0")
    if not isinstance(features, torch.Tensor):
        raise TypeError("features phải là torch.Tensor")
    if (
        features.ndim not in (2, 3)
        or min(features.shape[-2:]) == 0
        or (features.ndim == 3 and features.shape[0] == 0)
    ):
        raise ValueError(
            "features phải có shape (regions, feature_dim) hoặc "
            "(batch, regions, feature_dim), với các chiều phải > 0"
        )

    if tokenizer is None:
        tokenizer = load_gpt2_tokenizer()
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("tokenizer cần có bos_token_id và eos_token_id")

    single_image = features.ndim == 2
    feature_batch = features.unsqueeze(0) if single_image else features

    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration):
        parameter = None

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
            _diverse_search_one(
                model=model,
                features=image_features.unsqueeze(0),
                tokenizer=tokenizer,
                beam_size=beam_size,
                num_groups=num_groups,
                diversity_strength=diversity_strength,
                max_len=max_len,
                length_penalty=length_penalty,
            )
            for image_features in feature_batch
        ]
    finally:
        model.train(was_training)

    return captions[0] if single_image else captions


# Tên ngắn và tên tương ứng với greedy_decode/beam_decode trong cùng thư mục.
diverse_search = diverse_beam_search
diverse_beam_decode = diverse_beam_search
