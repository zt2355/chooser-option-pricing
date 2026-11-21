"""Small LSTM for volatility forecasting (Week 5 initial framework).

Deliberately modest: 1 layer, hidden 32, 40 epochs, CPU, fixed seeds —
the goal this week is a correct, reproducible pipeline; capacity and
tuning belong to Week 6.
"""

import numpy as np
import torch
from torch import nn


class VolLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def train_lstm(seq_train: np.ndarray, y_train: np.ndarray,
               eval_seqs: dict[str, np.ndarray], *, seed: int = 42,
               epochs: int = 40, lr: float = 1e-3,
               batch_size: int = 64) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"  # determinism over speed at this scale

    model = VolLSTM(seq_train.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(seq_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    n = len(Xt)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()

    model.eval()
    out = {}
    with torch.no_grad():
        out["train"] = model(Xt).numpy()
        for name, seq in eval_seqs.items():
            out[name] = model(torch.tensor(seq, dtype=torch.float32)).numpy()
    return out
