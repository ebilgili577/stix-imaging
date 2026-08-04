from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

NORMALIZATION_FACTOR = 4000.0
CHECKPOINT_FORMAT = "frozen_warning_head_v1"
EXPECTED_WARNING_LOGITS = ("unsupported", "outside_fov", "high_sidelobes")
# TODO: Adjust thresholds later if needed.
DEFAULT_CAUTION_THRESHOLD = 0.855
DEFAULT_HIGH_RISK_THRESHOLD = 0.9771
DEFAULT_FOV_THRESHOLD = 0.9967
DEFAULT_SIDELOBE_THRESHOLD = 0.7719


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

    def forward_with_embedding(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.net[:-1](x)
        prediction = self.net[-1](embedding)
        return prediction, embedding


class WarningHead(nn.Module):
    """Classifies unsupported locations from localizer features."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class LocalizerWithWarning(nn.Module):
    """Frozen flare localizer plus the FOV/sidelobe warning head."""

    def __init__(self, localizer: TorchMLP, head: WarningHead) -> None:
        super().__init__()
        self.localizer = localizer
        self.head = head
        self.localizer.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.localizer.eval()
        return self

    @torch.no_grad()
    def predict_event(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction, embedding = self.localizer.forward_with_embedding(x)
        logits = self.head(torch.cat([embedding, prediction], dim=1))
        probabilities = torch.sigmoid(logits)
        return {
            "prediction": prediction,
            "warning_probability": probabilities[:, 0],
            "outside_probability": probabilities[:, 1],
            "high_sidelobe_probability": probabilities[:, 2],
        }


def load_localizer_model(
    checkpoint: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[LocalizerWithWarning, dict[str, Any]]:
    """Load the bundled PyTorch localizer and warning head."""
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

    head_config = payload["warning_head"]
    logit_order = tuple(head_config.get("logit_order", ()))
    if logit_order != EXPECTED_WARNING_LOGITS:
        raise ValueError(
            f"Unexpected warning logit order {logit_order}; "
            f"expected {EXPECTED_WARNING_LOGITS}"
        )
    head = WarningHead(
        int(head_config["input_dim"]),
        int(head_config["hidden_dim"]),
        float(head_config["dropout"]),
    )
    head.load_state_dict(head_config["state_dict"])

    model = LocalizerWithWarning(localizer, head).to(device)
    model.eval()
    return model, payload


def predict_location(
    raw_counts: np.ndarray,
    model: LocalizerWithWarning,
    *,
    threshold_caution: float = DEFAULT_CAUTION_THRESHOLD,
    threshold_high_risk: float = DEFAULT_HIGH_RISK_THRESHOLD,
    threshold_fov: float = DEFAULT_FOV_THRESHOLD,
    threshold_sidelobe: float = DEFAULT_SIDELOBE_THRESHOLD,
) -> dict[str, object]:
    """Predict STIX location and FOV/sidelobe warnings from detector counts."""
    if model is None:
        raise RuntimeError("Localizer model not loaded")
    if not 0.0 <= threshold_caution <= threshold_high_risk <= 1.0:
        raise ValueError(
            "Warning thresholds must satisfy "
            "0 <= threshold_caution <= threshold_high_risk <= 1"
        )

    input_dim = model.localizer.input_dim
    if input_dim % 8:
        raise ValueError(f"Localizer input width {input_dim} is not divisible by 8")
    features = _counts_to_mlp_features(raw_counts, input_dim // 8)
    features = normalize(features).astype(np.float32)

    device = next(model.parameters()).device
    output = model.predict_event(torch.from_numpy(features).to(device))

    prediction = output["prediction"][0].cpu().numpy()
    x_arcsec, y_arcsec = denormalize(prediction).tolist()
    p_outside_fov = float(output["outside_probability"][0].item())
    p_high_sidelobes = float(output["high_sidelobe_probability"][0].item())
    p_unsupported_union = float(output["warning_probability"][0].item())

    fov_flag = p_outside_fov >= threshold_fov
    sidelobe_flag = p_high_sidelobes >= threshold_sidelobe
    warning = p_unsupported_union >= threshold_caution
    if p_unsupported_union >= threshold_high_risk:
        warning_status = "high_risk"
    elif warning:
        warning_status = "caution"
    else:
        warning_status = "no_warning"

    reasons = []
    if warning:
        if fov_flag:
            reasons.append("outside_fov")
        if sidelobe_flag:
            reasons.append("high_sidelobes")
        if not reasons:
            reasons.append("unsupported_localization")

    return {
        "status": "OK",
        "location_x_arcsec": x_arcsec,
        "location_y_arcsec": y_arcsec,
        "p_outside_fov": p_outside_fov,
        "p_high_sidelobes": p_high_sidelobes,
        "p_unsupported_union": p_unsupported_union,
        "warning": warning,
        "warning_status": warning_status,
        "reasons": reasons,
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
