"""
Module for extracting data and processing it for model input
"""

import astropy.units as u
import numpy as np
from astropy.time import Time
from stixpy.calibration.energy import get_sci_channels
from stixpy.calibration.visibility import (
    create_meta_pixels,
    create_visibility,
)
from sunpy.time import TimeRange

from .l1 import pixel_data_from_l1_json
from .schemas import Selection

# 1-based detector IDs in training column order (stix_train DETECTOR_ORDER).
# extract_counts reorders the 32-detector L1 array to this row order.
DETECTOR_ORDER = np.array(
    [
        3,
        20,
        22,
        16,
        14,
        32,
        21,
        26,
        4,
        24,
        8,
        28,
        15,
        27,
        31,
        6,
        30,
        2,
        25,
        5,
        23,
        7,
        29,
        1,
    ],
    dtype=int,
)


def extract_counts(
    l1_json: dict, selection: Selection
) -> tuple[np.ndarray, float]:
    """Sum big-pixel counts over the selection time/energy window.

    Returns
    -------
    counts : ndarray, shape (24, 8)
        Imaging detectors in DETECTOR_ORDER; cols 0–7 = top abcd + bottom abcd.
    selection_total_counts : float
        Sum of those counts (24 imaging detectors, selected time + energy).

    Each L1 box contributes if its integration interval overlaps the selection
    time window and each energy sub-bin lies within [e_channel_min, e_channel_max].
    """
    counts = np.zeros((32, 8), dtype=np.float32)
    for box in l1_json["boxes"]:
        if not box:
            continue
        if not _box_within_time(box, selection.start_unix, selection.end_unix):
            continue
        for energy_channel in box["counts"]:
            if not _in_energy_channel(
                energy_channel,
                selection.e_channel_min,
                selection.e_channel_max,
            ):
                continue
            # energy_channel[3] is the flat pixel count vector for that bin.
            counts += _flat_to_big_pixel_counts(
                np.asarray(energy_channel[3], dtype=np.float32),
            )
    imaging = counts[DETECTOR_ORDER - 1]
    return imaging, float(np.sum(imaging))


def extract_visibilities(l1_json: dict, selection: Selection):
    """Return uncalibrated stixpy Visibilities for the given selection.

    Builds a duck-typed pixel_data from the L1 JSON and delegates to
    stixpy create_meta_pixels (with no_shadowing=True, pixels='top+bot')
    and create_visibility — both called unmodified.
    """
    pixel_data = pixel_data_from_l1_json(l1_json)
    time_range = Time([selection.start_unix, selection.end_unix], format="unix")
    energy_range = _channels_to_energy_range(pixel_data, selection)
    meta = create_meta_pixels(
        pixel_data,
        time_range,
        energy_range,
        pixels="top+bot",
        no_shadowing=True,
    )
    vis = create_visibility(meta)
    t_center = TimeRange(vis.meta.time_range).center

    return vis, t_center


def _flat_to_big_pixel_counts(flat: np.ndarray) -> np.ndarray:
    """Map L1 count vector to (32, 8): top abcd + bottom abcd per detector."""
    n = flat.size
    if n == 384:
        # 12 pixels/det (big+small): keep big pixels in cols 0–7.
        return flat.reshape(32, 12)[:, :8]
    if n == 256:
        # 8 pixels/det (big only): already top abcd + bottom abcd.
        return flat.reshape(32, 8)
    raise ValueError(
        f"Unexpected L1 pixel count length {n} (expected 256 or 384)",
    )


def _channels_to_energy_range(pixel_data, selection: Selection):
    """Return [e_low, e_high] Quantity[keV] spanning the selection's channel range.

    Uses the stixpy science-channel table (date-dependent, internally cached)
    to convert integer channel indices to keV boundaries.
    """
    obstime = pixel_data.time_range.center
    sci_ch = get_sci_channels(obstime)
    ch_nums = np.asarray(sci_ch["Channel Number"])

    idx_lo = np.flatnonzero(ch_nums == selection.e_channel_min)
    idx_hi = np.flatnonzero(ch_nums == selection.e_channel_max)
    if not idx_lo.size or not idx_hi.size:
        raise ValueError(
            f"Science channel {selection.e_channel_min} or {selection.e_channel_max} "
            f"not found in channel table for {obstime}"
        )
    e_low = sci_ch["Elower"][idx_lo[0]]
    e_high = sci_ch["Eupper"][idx_hi[0]]
    return u.Quantity([e_low.to_value("keV"), e_high.to_value("keV")], u.keV)


def _box_within_time(box: dict, t_start: float, t_end: float) -> bool:
    """True if the box integration interval overlaps [t_start, t_end] (unix)."""
    half_dur = float(box["integrations"]) / 2.0
    tcenter = float(box["time"])
    return tcenter + half_dur >= t_start and tcenter - half_dur <= t_end


def _in_energy_channel(energy_channel, ch_min: int, ch_max: int) -> bool:
    """True if this L1 energy sub-bin lies fully inside [ch_min, ch_max]."""
    e1 = int(energy_channel[0])
    e2 = int(energy_channel[1])
    return e1 >= ch_min and e2 <= ch_max
