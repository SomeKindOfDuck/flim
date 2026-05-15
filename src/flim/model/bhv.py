import torch
from torch import nn


class GRURewardPredictor(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(input_size=z_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, z_seq):
        out, _ = self.rnn(z_seq)
        logits = self.fc(out)
        return torch.sigmoid(logits).squeeze(-1)


class LSTMRewardPredictor(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(input_size=z_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, z_seq):
        out, _ = self.rnn(z_seq)
        logits = self.fc(out)
        return torch.sigmoid(logits).squeeze(-1)


def reward_loss(pred: torch.Tensor, obs: torch.Tensor):
    from torch.nn.functional import binary_cross_entropy
    return binary_cross_entropy(pred, obs, reduction="sum")
