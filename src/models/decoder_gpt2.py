import torch
from torch import nn
from transformers import GPT2LMHeadModel

class GPT2Decoder(nn.Module):
    def __init__(self, feature_dim, gpt2_name="gpt2", dropout=0.1):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_name)
        
        self.img_proj = nn.Sequential(
            nn.Linear(feature_dim, self.gpt2.config.n_embd),
            nn.LayerNorm(self.gpt2.config.n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, features, input_ids, attention_mask=None):
        embeds = torch.cat([self.img_proj(features), self.gpt2.transformer.wte(input_ids)], dim=1)

        if attention_mask is not None:
            img_mask = torch.ones(features.shape[:2], dtype=attention_mask.dtype, device=features.device)
            attention_mask = torch.cat([img_mask, attention_mask], dim=1)

        return self.gpt2(inputs_embeds=embeds, attention_mask=attention_mask).logits[:, features.shape[1]:]