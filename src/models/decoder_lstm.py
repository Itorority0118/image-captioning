
import torch
from torch import nn


class AdditiveAttention(nn.Module):
    """
    Additive Attention:
    Tính trọng số chú ý giữa hidden state của LSTM
    và các vùng đặc trưng ảnh từ MobileNetV3-Small.
    """

    def __init__(self, feature_dim, hidden_dim, attention_dim):
        super().__init__()

        self.feature_proj = nn.Linear(feature_dim, attention_dim)
        self.hidden_proj = nn.Linear(hidden_dim, attention_dim)
        self.energy = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden):
        # features: (B, 49, 576)
        # hidden:   (B, 512)

        feature_keys = self.feature_proj(features)
        hidden_query = self.hidden_proj(hidden).unsqueeze(1)

        scores = self.energy(
            torch.tanh(feature_keys + hidden_query)
        ).squeeze(-1)

        attention_weights = torch.softmax(scores, dim=1)

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            features
        ).squeeze(1)

        return context, attention_weights


class LSTMDecoder(nn.Module):
    """
    MobileNetV3-Small + LSTM + Additive Attention.
    """

    def __init__(
        self,
        feature_dim,
        vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        attention_dim=256,
        dropout=0.3
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # GPT-2 BPE embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.dropout = nn.Dropout(dropout)

        # Additive Attention
        self.attention = AdditiveAttention(
            feature_dim,
            hidden_dim,
            attention_dim
        )

        # Initialize LSTM hidden state and cell state
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)

        # LSTM Decoder
        self.lstm = nn.LSTM(
            input_size=embedding_dim + feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Predict next GPT-2 token
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, input_ids):
        # features:  (B, 49, 576)
        # input_ids: (B, 39)

        global_features = features.mean(dim=1)

        hidden = torch.tanh(
            self.init_h(global_features)
        ).unsqueeze(0)

        cell = torch.tanh(
            self.init_c(global_features)
        ).unsqueeze(0)

        embeddings = self.dropout(
            self.embedding(input_ids)
        )

        outputs = []

        for t in range(input_ids.size(1)):
            context, _ = self.attention(
                features,
                hidden[-1]
            )

            lstm_input = torch.cat(
                [embeddings[:, t, :], context],
                dim=1
            ).unsqueeze(1)

            lstm_output, (hidden, cell) = self.lstm(
                lstm_input,
                (hidden, cell)
            )

            logits = self.output(
                self.dropout(lstm_output.squeeze(1))
            )

            outputs.append(logits)

        return torch.stack(outputs, dim=1)
