import torch
from torch import nn


class GRURewardPredictor(nn.Module):
    def __init__(self, z_dim, hidden_dim, num_layers=1, chrono_tmax=None):
        super().__init__()
        self.rnn = nn.GRU(input_size=z_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        if chrono_tmax is not None:
            chrono_init(self.rnn, chrono_tmax)

    def forward(self, z_seq):
        out, _ = self.rnn(z_seq)
        logits = self.fc(out)
        return torch.sigmoid(logits).squeeze(-1)


def chrono_init(rnn: nn.GRU, tmax: float):
    """
    更新ゲートのバイアスの初期値で、各ユニットの記憶の時定数を [1, tmax] ステップの一様分布にする(chrono initialization)
    PyTorch の GRU は h' = (1 - u) * n + u * h なので、時定数 T のユニットは u = 1 - 1 / T、つまりバイアス log(T - 1)
    ゲートの並びは (r, u, n)。学習でバイアスは変わるので、初期値として記憶の長さの目安を与えるだけ
    """
    H = rnn.hidden_size
    # 学習の乱数(データの順番、z のサンプリング)をずらさないよう、別の乱数生成器を使う
    g = torch.Generator().manual_seed(torch.initial_seed() % (2 ** 63))
    with torch.no_grad():
        for layer in range(rnn.num_layers):
            b_ih = getattr(rnn, f"bias_ih_l{layer}")
            b_hh = getattr(rnn, f"bias_hh_l{layer}")
            t = torch.empty(H).uniform_(1.0, float(tmax), generator=g)
            b_ih[H:2 * H] = torch.log(torch.clamp(t - 1.0, min=1e-3))
            b_hh[H:2 * H] = 0.0


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
