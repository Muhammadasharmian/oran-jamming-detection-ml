"""PyTorch neural network detectors with a scikit-learn compatible interface.

Architectures
-------------
``FeedForwardNet``
    Fully connected network. Used for the compact MLP (64-32) and the deeper
    MLP (128-64-32 with batch normalisation and heavier dropout).
``CNNLSTMNet``
    1-D convolutional feature extractor over the feature vector followed by
    an LSTM and a classification head. Used for the compact CNN-LSTM and the
    two-stage, bidirectional, two-layer variant.

Both are trained through :class:`TorchClassifier`, which handles label
encoding, optional class-balanced loss, mini-batching and inference, and
works for binary as well as multi-class problems.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from jamming_detection import config


class FeedForwardNet(nn.Module):
    """Multi-layer perceptron for tabular KPM features."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float | Sequence[float] = 0.2,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        n_hidden = len(hidden_dims)
        # Dropout / batch norm are applied after every hidden layer but the last.
        if isinstance(dropout, (int, float)):
            dropout = [float(dropout)] * max(n_hidden - 1, 0)
        layers: list[nn.Module] = []
        prev = input_dim
        for i, width in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, width))
            if batch_norm and i < n_hidden - 1:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            if i < n_hidden - 1 and i < len(dropout) and dropout[i] > 0:
                layers.append(nn.Dropout(dropout[i]))
            prev = width
        layers.append(nn.Linear(prev, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CNNLSTMNet(nn.Module):
    """Convolutional front end followed by an LSTM over the feature axis."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        conv_channels: Sequence[int] = (16,),
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        conv: list[nn.Module] = []
        in_ch = 1
        for out_ch in conv_channels:
            conv += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(out_ch),
            ]
            in_ch = out_ch
        if input_dim >= 2:
            conv.append(nn.MaxPool1d(kernel_size=2))
        self.cnn = nn.Sequential(*conv)

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.2 if lstm_layers > 1 else 0.0,
        )
        self.bidirectional = bidirectional
        head_in = lstm_hidden * (2 if bidirectional else 1)
        if bidirectional:
            self.head = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(head_in, 64), nn.ReLU(), nn.Linear(64, n_classes)
            )
        else:
            self.head = nn.Linear(head_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x.unsqueeze(1))  # (batch, channels, seq)
        x = x.permute(0, 2, 1)  # (batch, seq, channels)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1) if self.bidirectional else h_n[-1]
        return self.head(h)


_ARCHITECTURES = {"mlp": FeedForwardNet, "cnn_lstm": CNNLSTMNet}


class TorchClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn style trainer for the PyTorch architectures above.

    Args:
        architecture: ``"mlp"`` or ``"cnn_lstm"``.
        arch_params: Keyword arguments forwarded to the network constructor.
        epochs, batch_size, lr: Optimisation settings (Adam).
        class_weighted: Use inverse-frequency class weights in the loss.
        device: ``"cpu"``, ``"cuda"`` or ``None`` for automatic selection.
        random_state: Seed for weight initialisation and batch shuffling.
    """

    def __init__(
        self,
        architecture: str = "mlp",
        arch_params: Optional[Dict[str, Any]] = None,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        class_weighted: bool = True,
        device: Optional[str] = None,
        random_state: int = config.RANDOM_STATE,
        verbose: bool = False,
    ) -> None:
        self.architecture = architecture
        self.arch_params = arch_params
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.class_weighted = class_weighted
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> TorchClassifier:
        if self.architecture not in _ARCHITECTURES:
            raise ValueError(f"Unknown architecture '{self.architecture}'")
        torch.manual_seed(self.random_state)
        generator = torch.Generator().manual_seed(self.random_state)
        device = self._device()

        self.label_encoder_ = LabelEncoder()
        y_idx = self.label_encoder_.fit_transform(np.asarray(y))
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)
        X = np.asarray(X, dtype=np.float32)
        self.n_features_in_ = X.shape[1]

        net_cls = _ARCHITECTURES[self.architecture]
        self.model_ = net_cls(X.shape[1], n_classes, **(self.arch_params or {})).to(device)

        weight = None
        if self.class_weighted:
            counts = np.bincount(y_idx, minlength=n_classes).astype(np.float32)
            weight = torch.tensor(len(y_idx) / (n_classes * np.maximum(counts, 1)), device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y_idx.astype(np.int64))),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=len(X) > self.batch_size,  # avoid size-1 batches with BatchNorm
        )

        self.loss_history_: list[float] = []
        self.model_.train()
        for epoch in range(self.epochs):
            running, n_seen = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * len(xb)
                n_seen += len(xb)
            self.loss_history_.append(running / max(n_seen, 1))
            if self.verbose:
                print(f"  epoch {epoch + 1:3d}/{self.epochs}  loss {self.loss_history_[-1]:.4f}")
        self.model_.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        device = next(self.model_.parameters()).device
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        out = []
        for start in range(0, len(X_t), 4096):
            logits = self.model_(X_t[start : start + 4096].to(device))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, len(self.classes_)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        if "model_" in state:  # store weights on CPU so models load anywhere
            state["model_"] = state["model_"].to("cpu")
        return state


def _from_preset(architecture: str, preset: Dict[str, Any], **overrides: Any) -> TorchClassifier:
    preset = {**preset, **overrides}
    train_keys = {"epochs", "batch_size", "lr", "class_weighted", "device", "random_state", "verbose"}
    train = {k: v for k, v in preset.items() if k in train_keys}
    arch = {k: v for k, v in preset.items() if k not in train_keys}
    return TorchClassifier(architecture=architecture, arch_params=arch, **train)


def build_mlp(**overrides: Any) -> TorchClassifier:
    """Compact two-hidden-layer MLP (64-32)."""
    return _from_preset("mlp", config.MLP, **overrides)


def build_deep_mlp(**overrides: Any) -> TorchClassifier:
    """Three-hidden-layer MLP (128-64-32) with batch normalisation."""
    return _from_preset("mlp", config.DEEP_MLP, **overrides)


def build_cnn_lstm(**overrides: Any) -> TorchClassifier:
    """Single Conv1d stage with a unidirectional LSTM."""
    return _from_preset("cnn_lstm", config.CNN_LSTM, **overrides)


def build_bi_cnn_lstm(**overrides: Any) -> TorchClassifier:
    """Two Conv1d stages with a two-layer bidirectional LSTM."""
    return _from_preset("cnn_lstm", config.BI_CNN_LSTM, **overrides)
