from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

NORMALIZATION_FACTOR = 4000.0
CHECKPOINT_FORMAT = "frozen_warning_head_v1"


class TorchMLP(nn.Module):
    """Location network architecture stored in the PyTorch checkpoint."""

    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.input_dim = input_dim
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_localizer_model(
    checkpoint: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[TorchMLP, dict[str, Any]]:
    """Load the PyTorch MLP from the checkpoint."""
    checkpoint = Path(checkpoint)
    device = torch.device(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{checkpoint} is not a {CHECKPOINT_FORMAT} checkpoint")

    localizer_config = payload["localizer"]
    localizer = TorchMLP(
        int(localizer_config["input_shape"][0]),
        list(localizer_config["hidden_dims"]),
    )
    localizer.load_state_dict(localizer_config["state_dict"])
    localizer.to(device)
    localizer.eval()
    return localizer, payload


def predict_location(
    raw_counts: np.ndarray,
    model: TorchMLP,
) -> dict[str, object]:
    """Predict the STIX location from detector counts."""
    if model is None:
        raise RuntimeError("Localizer model not loaded")

    input_dim = model.input_dim
    if input_dim % 8:
        raise ValueError(f"Localizer input width {input_dim} is not divisible by 8")
    features = _counts_to_mlp_features(raw_counts, input_dim // 8)
    features = normalize(features).astype(np.float32)

    device = next(model.parameters()).device
    with torch.no_grad():
        prediction = model(torch.from_numpy(features).to(device)).cpu().numpy()

    x_arcsec, y_arcsec = denormalize(prediction[0]).tolist()

    return {
        "status": "OK",
        "location_x_arcsec": x_arcsec,
        "location_y_arcsec": y_arcsec,
    }


def normalize(x: np.ndarray) -> np.ndarray:
    """Per-event max-normalize localizer features, safely handling all zeros."""
    row_max = x.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    return x / row_max


def denormalize(x: np.ndarray) -> np.ndarray:
    """Undo the training target normalization and return arcseconds."""
    return x * NORMALIZATION_FACTOR


def _counts_to_mlp_features(raw_counts: np.ndarray, detector_count: int) -> np.ndarray:
    """Convert ordered detector rows to the checkpoint's feature-column order."""
    raw_counts = np.asarray(raw_counts)
    expected_shape = (detector_count, 8)
    if raw_counts.ndim != 2 or raw_counts.shape[0] < detector_count:
        raise ValueError(
            f"Expected at least {expected_shape} detector counts, got {raw_counts.shape}"
        )
    if raw_counts.shape[1] != expected_shape[1]:
        raise ValueError(f"Expected detector rows with 8 counts, got {raw_counts.shape}")

    counts = raw_counts[:detector_count]
    # L1 rows: top_a,b,c,d | bot_a,b,c,d. Checkpoint columns:
    # a_top,a_bot,b_top,b_bot,c_top,c_bot,d_top,d_bot.
    return counts[:, [0, 4, 1, 5, 2, 6, 3, 7]].reshape(1, -1)
