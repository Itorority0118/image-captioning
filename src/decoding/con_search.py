
import torch
import torch.nn.functional as F

from src.data.text_preprocess import SPECIAL_TOKEN_ID, decode_caption


def _get_gpt2_visual_attention(outputs, prefix_len, batch_index=0):
    """
    Extract visual attention from GPT-2.

    Average attention across all Transformer layers and heads.
    Keep only attention from the last text token to visual prefix tokens.
    Normalize the visual attention to sum to 1.
    """

    if outputs.attentions is None:
        raise RuntimeError(
            "GPT-2 did not return attention weights. "
            "Check output_attentions support in Transformers."
        )

    attentions = torch.stack(outputs.attentions, dim=0)

    visual_attention = attentions[
        :, batch_index, :, -1, :prefix_len
    ]

    visual_attention = visual_attention.mean(dim=(0, 1))

    visual_attention = visual_attention / (
        visual_attention.sum() + 1e-12
    )

    return visual_attention


def _gpt2_forward(
    model,
    features,
    input_ids,
    prefix,
    return_attention=False
):
    """
    Full-sequence forward for GPT-2 Decoder.

    This implementation intentionally does not modify GPT2Decoder.
    It directly uses the existing img_proj and gpt2 modules.
    """

    batch_size = input_ids.size(0)
    prefix_len = prefix.size(1)

    text_embeds = model.gpt2.transformer.wte(input_ids)

    if prefix.size(0) != batch_size:
        prefix = prefix.expand(batch_size, -1, -1)

    inputs_embeds = torch.cat(
        [prefix, text_embeds],
        dim=1
    )

    attention_mask = torch.ones(
        inputs_embeds.shape[:2],
        dtype=torch.long,
        device=input_ids.device
    )

    outputs = model.gpt2(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        output_attentions=return_attention,
        use_cache=False,
        return_dict=True
    )

    logits = outputs.logits[:, -1, :]

    hidden = outputs.hidden_states[-1][:, -1, :]

    attention_weights = None

    if return_attention:
        attention_weights = [
            _get_gpt2_visual_attention(
                outputs,
                prefix_len,
                batch_index=i
            )
            for i in range(batch_size)
        ]

    return logits, hidden, attention_weights


@torch.no_grad()
def contrastive_search(
    model,
    features,
    max_len=40,
    top_k=5,
    alpha=0.6,
    return_attention=False
):
    """
    Contrastive Search supporting:

    1. LSTM Decoder with forward_step() and Additive Attention.
    2. GPT-2 Decoder with Visual Prefix Conditioning.

    Args:
        model: LSTMDecoder or GPT2Decoder.
        features: (49, feature_dim) or (1, 49, feature_dim).
        max_len: Maximum generated caption length.
        top_k: Number of candidate tokens.
        alpha: Degeneration penalty coefficient.
        return_attention: Return Attention History.

    Returns:
        caption

        Or:
        (caption, attention_history)
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    if max_len < 2:
        raise ValueError(
            "max_len must be at least 2."
        )

    is_gpt2 = (
        hasattr(model, "gpt2")
        and hasattr(model, "img_proj")
    )

    is_lstm = (
        hasattr(model, "forward_step")
        and hasattr(model, "attention")
    )

    if not is_gpt2 and not is_lstm:
        raise TypeError(
            "Unsupported Decoder. Expected LSTMDecoder "
            "or GPT2Decoder."
        )

    was_training = model.training
    model.eval()

    try:

        device = next(model.parameters()).device

        if features.dim() == 2:
            features = features.unsqueeze(0)

        if features.dim() != 3 or features.size(0) != 1:
            raise ValueError(
                "Contrastive Search expects one image: "
                "(49, feature_dim) or (1, 49, feature_dim)."
            )

        features = features.to(
            device=device,
            dtype=torch.float32
        )

        input_ids = torch.tensor(
            [[SPECIAL_TOKEN_ID]],
            dtype=torch.long,
            device=device
        )

        hidden_history = []
        attention_history = []

        if is_gpt2:

            prefix = model.img_proj(features)

            logits, hidden, current_attention = _gpt2_forward(
                model,
                features,
                input_ids,
                prefix,
                return_attention=return_attention
            )

            hidden_history.append(hidden)

            for _ in range(max_len - 1):

                probabilities = torch.softmax(
                    logits.float(),
                    dim=-1
                )

                candidate_count = min(
                    top_k,
                    probabilities.size(-1)
                )

                candidate_probs, candidate_ids = torch.topk(
                    probabilities,
                    k=candidate_count,
                    dim=-1
                )

                candidate_ids = candidate_ids.squeeze(0)

                candidate_prefix = prefix.expand(
                    candidate_count, -1, -1
                )

                candidate_context = input_ids.expand(
                    candidate_count, -1
                )

                candidate_inputs = torch.cat(
                    [
                        candidate_context,
                        candidate_ids.unsqueeze(1)
                    ],
                    dim=1
                )

                (
                    candidate_logits,
                    candidate_representations,
                    candidate_attention
                ) = _gpt2_forward(
                    model,
                    features,
                    candidate_inputs,
                    candidate_prefix,
                    return_attention=return_attention
                )

                previous_representations = torch.cat(
                    hidden_history,
                    dim=0
                )

                similarity = F.cosine_similarity(
                    candidate_representations.unsqueeze(1),
                    previous_representations.unsqueeze(0),
                    dim=-1
                )

                degeneration_penalty = similarity.max(
                    dim=1
                ).values

                scores = (
                    candidate_probs.squeeze(0)
                    - alpha * degeneration_penalty
                )

                best_index = scores.argmax()

                next_token = candidate_ids[
                    best_index
                ].view(1, 1)

                if next_token.item() == SPECIAL_TOKEN_ID:
                    break

                if return_attention:
                    attention_history.append(
                        current_attention[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                input_ids = torch.cat(
                    [input_ids, next_token],
                    dim=1
                )

                hidden = candidate_representations[
                    best_index:best_index + 1
                ]

                hidden_history.append(hidden)

                logits = candidate_logits[
                    best_index:best_index + 1
                ]


                if return_attention:
                    current_attention = [
                        candidate_attention[best_index]
                    ]


        else:

            feature_keys = model.attention.feature_proj(
                features
            )

            # Initial decoding step.
            if return_attention:

                (
                    logits,
                    hidden,
                    cell,
                    feature_keys,
                    attention_weights
                ) = model.forward_step(
                    features,
                    input_ids,
                    feature_keys=feature_keys,
                    return_attention=True
                )

            else:

                (
                    logits,
                    hidden,
                    cell,
                    feature_keys
                ) = model.forward_step(
                    features,
                    input_ids,
                    feature_keys=feature_keys
                )

            hidden_history = [
                hidden.squeeze(0)
            ]

            for _ in range(max_len - 1):

                probabilities = torch.softmax(
                    logits.float(),
                    dim=-1
                )

                candidate_count = min(
                    top_k,
                    probabilities.size(-1)
                )

                candidate_probs, candidate_ids = torch.topk(
                    probabilities,
                    k=candidate_count,
                    dim=-1
                )

                candidate_ids = candidate_ids.squeeze(0)

                candidate_features = features.expand(
                    candidate_count, -1, -1
                )

                candidate_feature_keys = feature_keys.expand(
                    candidate_count, -1, -1
                )

                candidate_hidden = hidden.expand(
                    -1, candidate_count, -1
                ).contiguous()

                candidate_cell = cell.expand(
                    -1, candidate_count, -1
                ).contiguous()

                candidate_inputs = candidate_ids.unsqueeze(1)

                (
                    candidate_logits,
                    next_hidden,
                    next_cell,
                    _
                ) = model.forward_step(
                    candidate_features,
                    candidate_inputs,
                    hidden=candidate_hidden,
                    cell=candidate_cell,
                    feature_keys=candidate_feature_keys
                )

                candidate_representations = (
                    next_hidden.squeeze(0)
                )

                previous_representations = torch.cat(
                    hidden_history,
                    dim=0
                )

                similarity = F.cosine_similarity(
                    candidate_representations.unsqueeze(1),
                    previous_representations.unsqueeze(0),
                    dim=-1
                )

                degeneration_penalty = similarity.max(
                    dim=1
                ).values

                scores = (
                    candidate_probs.squeeze(0)
                    - alpha * degeneration_penalty
                )

                best_index = scores.argmax()

                next_token = candidate_ids[
                    best_index
                ].view(1, 1)

                if next_token.item() == SPECIAL_TOKEN_ID:
                    break

                if return_attention:
                    attention_history.append(
                        attention_weights[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                input_ids = torch.cat(
                    [input_ids, next_token],
                    dim=1
                )

                hidden = next_hidden[
                    :, best_index:best_index + 1, :
                ]

                cell = next_cell[
                    :, best_index:best_index + 1, :
                ]

                logits = candidate_logits[
                    best_index:best_index + 1
                ]

                hidden_history.append(
                    hidden.squeeze(0)
                )

                if return_attention:
                    _, attention_weights = model.attention(
                        features,
                        hidden[-1],
                        feature_keys=feature_keys
                    )

        caption = decode_caption(input_ids[0])

        if return_attention:
            return caption, attention_history

        return caption

    finally:
        model.train(was_training)
