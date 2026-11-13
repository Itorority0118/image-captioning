
import torch
import torch.nn.functional as F

from src.data.text_preprocess import SPECIAL_TOKEN_ID, decode_caption


@torch.no_grad()
def contrastive_search(
    model,
    features,
    max_len=40,
    top_k=5,
    alpha=0.6,
    return_attention=False
):

    if not hasattr(model, "forward_step"):
        raise AttributeError(
            "Model must implement forward_step()."
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if max_len < 2:
        raise ValueError("max_len must be at least 2.")

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

        feature_keys = model.attention.feature_proj(features)

        # Initial decoding step
        if return_attention:
            logits, hidden, cell, feature_keys, attention_weights = (
                model.forward_step(
                    features,
                    input_ids,
                    feature_keys=feature_keys,
                    return_attention=True
                )
            )
        else:
            logits, hidden, cell, feature_keys = model.forward_step(
                features,
                input_ids,
                feature_keys=feature_keys
            )

        hidden_history = [hidden.squeeze(0)]
        attention_history = []

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

            # Evaluate all candidate tokens
            candidate_logits, next_hidden, next_cell, _ = (
                model.forward_step(
                    candidate_features,
                    candidate_inputs,
                    hidden=candidate_hidden,
                    cell=candidate_cell,
                    feature_keys=candidate_feature_keys
                )
            )

            candidate_representations = next_hidden.squeeze(0)

            previous_representations = torch.cat(
                hidden_history,
                dim=0
            )

            similarity = F.cosine_similarity(
                candidate_representations.unsqueeze(1),
                previous_representations.unsqueeze(0),
                dim=-1
            )

            degeneration_penalty = similarity.max(dim=1).values

            # Original Contrastive Search scoring
            scores = (
                candidate_probs.squeeze(0)
                - alpha * degeneration_penalty
            )

            best_index = scores.argmax()

            next_token = candidate_ids[best_index].view(1, 1)

            # Stop before recording EOS as a caption token
            if next_token.item() == SPECIAL_TOKEN_ID:
                break

            # Save attention used to predict the selected token
            if return_attention:
                attention_history.append(
                    attention_weights[0].detach().cpu().numpy()
                )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1
            )

            # Keep the selected candidate's states
            hidden = next_hidden[:, best_index:best_index + 1, :]
            cell = next_cell[:, best_index:best_index + 1, :]

            logits = candidate_logits[best_index:best_index + 1]

            hidden_history.append(
                hidden.squeeze(0)
            )

            # Calculate attention for the next decoding step
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
