import numpy as np

NORMALIZATION_FACTOR = 4000.0

def predict_location(raw_counts: np.ndarray, mlp) -> dict:
    """MLP location from ordered detector counts → STIX (x, y) arcsec.

    Uses the first ``input_shape[1] // 8`` detectors (e.g. 9 → 72 features),
    per-event max-normalize, then denormalize network output by ×4000.
    """
    if mlp is None:
        raise RuntimeError('MLP model not loaded')
    col_count = mlp.input_shape[1] // 8
    X = _counts_to_mlp_features(raw_counts, col_count)
    X = normalize(X)
    preds = mlp.predict(X, verbose=0)
    x_arcsec, y_arcsec = (denormalize(preds[0])).tolist()
    return {
        'status': 'OK',
        'location_x_arcsec': float(x_arcsec),
        'location_y_arcsec': float(y_arcsec),
    }



def normalize(X: np.ndarray) -> np.ndarray:
    """Per-event max-normalize MLP features (rows)."""
    return X / X.max(axis=1, keepdims=True)


def denormalize(X: np.ndarray) -> np.ndarray:
    """Scale MLP network output back to arcsec (training used / 4000)."""
    return X * NORMALIZATION_FACTOR


def _counts_to_mlp_features(raw_counts: np.ndarray, col_count: int) -> np.ndarray:
    """(col_count, 8) → (1, col_count * 8) in training column order.

    Per detector, L1 layout is top_a,b,c,d | bot_a,b,c,d. Training columns are
    a_top, a_bot, b_top, b_bot, … for each detector in DETECTOR_ORDER.
    """
    features = []
    for i in range(col_count):
        r = raw_counts[i]
        # row layout: top_a,b,c,d | bot_a,b,c,d  →  a_top,a_bot,b_top,b_bot,...
        features.extend([r[0], r[4], r[1], r[5], r[2], r[6], r[3], r[7]])
    return np.array(features, dtype=np.float64).reshape(1, -1)

