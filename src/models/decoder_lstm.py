import torch
from torch import nn

class AdditiveAttention(nn.Module):

    def __init__(self, feature_dim, hidden_dim, attention_dim):
        super().__init__()

        self.feature_proj = nn.Linear(feature_dim, attention_dim)
        self.hidden_proj = nn.Linear(hidden_dim, attention_dim)
        self.energy = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden, feature_keys=None):

        if feature_keys is None:
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

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.dropout = nn.Dropout(dropout)

        self.attention = AdditiveAttention(
            feature_dim,
            hidden_dim,
            attention_dim
        )

        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim + feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output = nn.Linear(hidden_dim, vocab_size)


    
    def forward_step(
        self,
        features,
        input_ids,
        hidden=None,
        cell=None,
        feature_keys=None,
        return_attention=False
    ):

        if hidden is None or cell is None:
            global_features = features.mean(dim=1)

            hidden = torch.tanh(
                self.init_h(global_features)
            ).unsqueeze(0)

            cell = torch.tanh(
                self.init_c(global_features)
            ).unsqueeze(0)

        if feature_keys is None:
            feature_keys = self.attention.feature_proj(features)

        embedding = self.dropout(
            self.embedding(input_ids)
        )

        context, attention_weights = self.attention(
            features,
            hidden[-1],
            feature_keys=feature_keys
        )

        lstm_input = torch.cat(
            [embedding[:, 0, :], context],
            dim=1
        ).unsqueeze(1)

        lstm_output, (hidden, cell) = self.lstm(
            lstm_input,
            (hidden, cell)
        )

        hidden_representation = self.dropout(
            lstm_output.squeeze(1)
        )

        logits = self.output(hidden_representation)

        if return_attention:
            return logits, hidden, cell, feature_keys, attention_weights

        return logits, hidden, cell, feature_keys


    def forward(self, features, input_ids):

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

        feature_keys = self.attention.feature_proj(features)

        outputs = []

        for t in range(input_ids.size(1)):
            context, _ = self.attention(
                features,
                hidden[-1],
                feature_keys=feature_keys
            )

            lstm_input = torch.cat(
                [embeddings[:, t, :], context],
                dim=1
            ).unsqueeze(1)

            lstm_output, (hidden, cell) = self.lstm(
                lstm_input,
                (hidden, cell)
            )

            outputs.append(
                self.dropout(lstm_output.squeeze(1))
            )

        hidden_sequence = torch.stack(outputs, dim=1)

        logits = self.output(hidden_sequence)

        return logits
