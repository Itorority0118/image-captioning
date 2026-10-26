import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class GPT2Decoder(nn.Module):
    def __init__(self, feature_dim, gpt2_model_name="gpt2", dropout=0.1):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.hidden_size = self.gpt2.config.n_embd  # 768 cho gpt2 base

        self.img_proj = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, features, input_ids, attention_mask=None):
        """
        features: (batch_size, num_patches, feature_dim)
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len) - 1 cho token thật, 0 cho padding
        """
        batch_size, num_patches, _ = features.shape

        img_embeds = self.img_proj(features)

        text_embeds = self.gpt2.transformer.wte(input_ids)

        combined_embeds = torch.cat([img_embeds, text_embeds], dim=1)

        if attention_mask is not None:
            img_mask = torch.ones((batch_size, num_patches), dtype=attention_mask.dtype, device=attention_mask.device)
            combined_mask = torch.cat([img_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        outputs = self.gpt2(inputs_embeds=combined_embeds, attention_mask=combined_mask)

        text_logits = outputs.logits[:, num_patches:, :]
        return text_logits