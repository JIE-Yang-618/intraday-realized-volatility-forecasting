from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseForecaster, FitResult

class _MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class MLPForecaster(BaseForecaster):
    name = "mlp"

    def __init__(self, hidden: int = 64, dropout: float = 0.1, lr: float = 1e-3, epochs: int = 10, batch_size: int = 128) -> None:
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model: _MLP | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model = _MLP(d_in=X.shape[1], hidden=self.hidden, dropout=self.dropout).to(device)
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

        return FitResult(params={"hidden": self.hidden, "dropout": self.dropout})

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
        return pred.astype(np.float32)
