from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseForecaster, FitResult

class _RNNRegressor(nn.Module):
    def __init__(self, cell: str, d_in: int, hidden: int, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        if cell.lower() == "lstm":
            self.rnn = nn.LSTM(d_in, hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        elif cell.lower() == "gru":
            self.rnn = nn.GRU(d_in, hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        else:
            raise ValueError(f"Unknown cell: {cell}")
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

def _make_sequences(X_lag: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert lag matrix (n, p) into sequences (n-seq_len+1, seq_len, 1)."""
    n, p = X_lag.shape
    # treat each row as time-ordered lags; for demo we interpret first columns as most recent lags
    # create rolling windows along lag dimension
    m = p - seq_len + 1
    seq = np.zeros((n, seq_len, m), dtype=np.float32)
    for j in range(m):
        seq[:, :, j] = X_lag[:, j:j+seq_len]
    # collapse feature dimension as channels
    # final: (n*m, seq_len, 1)
    seq2 = seq.transpose(0, 2, 1).reshape(n * m, seq_len, 1)
    return seq2, np.repeat(np.arange(n), m)

class RNNForecaster(BaseForecaster):
    name = "rnn"

    def __init__(
        self,
        cell: str = "gru",
        hidden: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 128,
        seq_len: int = 20,
    ) -> None:
        self.cell = cell
        self.hidden = hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.model: _RNNRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        # X is lag matrix (n, p). Build sequences over lag dimension (demo)
        seq, idx = _make_sequences(X, self.seq_len)
        # align y with repeated indices (use the original y)
        y_rep = y[idx].astype(np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ds = TensorDataset(torch.tensor(seq, dtype=torch.float32), torch.tensor(y_rep, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model = _RNNRegressor(self.cell, d_in=1, hidden=self.hidden, num_layers=self.num_layers, dropout=self.dropout).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

        return FitResult(params={"cell": self.cell, "hidden": self.hidden, "seq_len": self.seq_len})

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        seq, idx = _make_sequences(X, self.seq_len)
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.tensor(seq, dtype=torch.float32).to(device)).cpu().numpy()
        # average repeated predictions back to original sample index
        n = X.shape[0]
        out = np.zeros(n, dtype=np.float32)
        cnt = np.zeros(n, dtype=np.float32)
        for k, i in enumerate(idx):
            out[i] += pred[k]
            cnt[i] += 1.0
        out = out / np.maximum(cnt, 1.0)
        return out.astype(np.float32)
