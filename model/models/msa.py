from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseForecaster, FitResult

class _MSAEncoder(nn.Module):
    """Multi-Scale Attention encoder (demo).

    Implements:
    - causal 1D convolutions over lag sequence at multiple calendar horizons H in {3,5,10} days
    - attention across horizons to build a multi-scale descriptor
    - recurrent backbone (LSTM/GRU) on the enhanced sequence
    """
    def __init__(
        self,
        cell: str,
        horizons_days: List[int],
        obs_per_day: int,
        conv_channels: int,
        attn_size: int,
        hidden: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.horizons_days = horizons_days
        self.obs_per_day = obs_per_day
        self.convs = nn.ModuleList()
        for H in horizons_days:
            k = max(3, H * obs_per_day)  # kernel length scaled by sampling interval
            self.convs.append(nn.Conv1d(1, conv_channels, kernel_size=k, padding=k-1))  # causal-ish (trim later)
        self.attn_W = nn.Linear(conv_channels, attn_size)
        self.attn_v = nn.Linear(attn_size, 1, bias=False)

        rnn_in = 1 + conv_channels  # original lag value + msa descriptor channel
        if cell.lower() == "lstm":
            self.rnn = nn.LSTM(rnn_in, hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        else:
            self.rnn = nn.GRU(rnn_in, hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1) lag-sequence
        B, T, _ = x.shape
        # build conv features on (B, 1, T)
        x1 = x.transpose(1, 2)  # (B,1,T)
        hs = []
        for conv in self.convs:
            z = conv(x1)  # (B,C,T + pad)
            z = z[:, :, :T]  # trim to causal alignment
            hs.append(z.mean(dim=2))  # (B,C) horizon summary (demo)
        H = torch.stack(hs, dim=1)  # (B, nH, C)

        # attention across horizons
        a = torch.tanh(self.attn_W(H))      # (B, nH, attn_size)
        u = self.attn_v(a).squeeze(-1)      # (B, nH)
        alpha = torch.softmax(u, dim=1)     # (B, nH)
        h_ms = (alpha.unsqueeze(-1) * H).sum(dim=1)  # (B, C)

        # expand descriptor over time and concatenate
        h_ms_seq = h_ms.unsqueeze(1).repeat(1, T, 1)         # (B,T,C)
        x_enh = torch.cat([x, h_ms_seq], dim=2)              # (B,T,1+C)

        out, _ = self.rnn(x_enh)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

class MSAForecaster(BaseForecaster):
    name = "msa"

    def __init__(
        self,
        cell: str = "gru",
        horizons_days: Optional[List[int]] = None,
        obs_per_day: int = 24,
        conv_channels: int = 8,
        attn_size: int = 16,
        hidden: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 128,
        seq_len: int = 60,
    ) -> None:
        self.cell = cell
        self.horizons_days = horizons_days or [3, 5, 10]
        self.obs_per_day = obs_per_day
        self.conv_channels = conv_channels
        self.attn_size = attn_size
        self.hidden = hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.model: _MSAEncoder | None = None

    def _seq_from_lags(self, X: np.ndarray) -> np.ndarray:
        # Use first seq_len lags as sequence (most recent first) for demo
        T = min(self.seq_len, X.shape[1])
        seq = X[:, :T].astype(np.float32)
        return seq.reshape(seq.shape[0], T, 1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq = self._seq_from_lags(X)
        ds = TensorDataset(torch.tensor(seq, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model = _MSAEncoder(
            cell=self.cell,
            horizons_days=self.horizons_days,
            obs_per_day=self.obs_per_day,
            conv_channels=self.conv_channels,
            attn_size=self.attn_size,
            hidden=self.hidden,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(device)

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

        return FitResult(params={
            "cell": self.cell,
            "horizons_days": self.horizons_days,
            "obs_per_day": self.obs_per_day,
            "conv_channels": self.conv_channels,
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        device = next(self.model.parameters()).device
        seq = self._seq_from_lags(X)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.tensor(seq, dtype=torch.float32).to(device)).cpu().numpy()
        return pred.astype(np.float32)
