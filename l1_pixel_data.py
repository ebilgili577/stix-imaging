"""Build a duck-typed pixel_data object from an L1 JSON dict.

The returned SimpleNamespace satisfies the exact interface consumed by
stixpy.calibration.visibility.create_meta_pixels (and its internal
get_elut_correction helper) so that all physics code (livetime, ELUT edge
correction, rate/keV/cm² normalisation) runs through unmodified stixpy.

Attributes on the returned object
----------------------------------
times : Time, shape (n_t,)
duration : Quantity[s], shape (n_t,)
data : QTable with columns
    counts          – Quantity[ct],   shape (n_t, 32, 12, n_e)
    counts_comp_err – Quantity[ct],   shape (n_t, 32, 12, n_e), zeros
    triggers        – float ndarray,  shape (n_t, 16)
    timedel         – Quantity[s],    shape (n_t,)
    rcr             – int ndarray,    shape (n_t,)
    pixel_masks     – int ndarray,    shape (n_t,)  [constant]
    detector_masks  – int ndarray,    shape (n_t,)  [constant]
energies : QTable with columns e_low, e_high [keV], n_e rows
energy_masks : SimpleNamespace with .energy_mask : bool ndarray shape (32,)
time_range : sunpy TimeRange
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import astropy.units as u
from astropy.table import QTable
from astropy.time import Time
from sunpy.time import TimeRange

from stixpy.calibration.energy import get_sci_channels


def pixel_data_from_l1_json(l1_json: dict) -> SimpleNamespace:
    """Return a SimpleNamespace compatible with stixpy create_meta_pixels.

    Parameters
    ----------
    l1_json:
        Decoded L1 compressed-pixel-data JSON as returned by the SDC.

    Returns
    -------
    SimpleNamespace
        Duck-typed pixel_data object accepted by stixpy create_meta_pixels.
    """
    boxes = [b for b in l1_json['boxes'] if 'time' in b and 'counts' in b]
    if not boxes:
        raise ValueError('No usable time boxes found in L1 JSON')

    n_t = len(boxes)

    # ------------------------------------------------------------------
    # Energy channel setup
    # energy_bins: [[ch, ch], ...] — single-channel bins only
    # Channel numbers are 0-based STIX science channel indices.
    # ------------------------------------------------------------------
    energy_bins_raw = l1_json['energy_bins']
    for e1, e2 in energy_bins_raw:
        if int(e1) != int(e2):
            raise ValueError(
                f'Multi-channel energy bin [{e1}, {e2}] is not supported; '
                'only single-channel bins (e1 == e2) are handled.'
            )
    chan_nums = [int(e[0]) for e in energy_bins_raw]
    n_e = len(chan_nums)
    chan_to_idx: dict[int, int] = {ch: idx for idx, ch in enumerate(chan_nums)}

    # Nominal energy edges in keV from the stixpy science-channel table.
    # The table is date-dependent and internally cached by stixpy.
    first_time = Time(float(boxes[0]['time']), format='unix')
    sci_ch = get_sci_channels(first_time)

    ch_arr = np.asarray(sci_ch['Channel Number'])
    sel_mask = np.isin(ch_arr, chan_nums)
    sel_sci = sci_ch[sel_mask]

    energies = QTable({
        'e_low': sel_sci['Elower'],
        'e_high': sel_sci['Eupper'],
    })

    # energy_mask: shape (32,) bool
    # Marks which of the 32 STIX science channels are present in this data.
    # Used by stixpy get_elut_correction to slice the ELUT along the energy axis.
    energy_mask = np.zeros(32, dtype=bool)
    for ch in chan_nums:
        energy_mask[ch] = True

    # ------------------------------------------------------------------
    # Per-box data arrays
    # ------------------------------------------------------------------
    times_unix = np.array([float(b['time']) for b in boxes])
    integrations = np.array([float(b['integrations']) for b in boxes])
    rcr_arr = np.array([int(b.get('rcr', 0)) for b in boxes])
    triggers_arr = np.array([b['triggers'] for b in boxes], dtype=np.float64)  # (n_t, 16)

    # counts: (n_t, 32 detectors, 12 pixels, n_e energy channels) in ct.
    # L1 JSON delivers 8 big pixels per detector (256-element flat) or
    # 12 pixels (384-element flat). Big pixels fill columns 0–7 of dim 2.
    counts_raw = np.zeros((n_t, 32, 12, n_e), dtype=np.float64)
    for t_i, box in enumerate(boxes):
        for entry in box['counts']:
            ch = int(entry[0])
            if ch not in chan_to_idx:
                continue
            e_idx = chan_to_idx[ch]
            flat = np.asarray(entry[3], dtype=np.float64)
            n = flat.size
            if n == 384:
                counts_raw[t_i, :, :, e_idx] = flat.reshape(32, 12)
            elif n == 256:
                # 8 big pixels only — scatter into columns 0–7; cols 8–11 stay zero.
                counts_raw[t_i, :, :8, e_idx] = flat.reshape(32, 8)
            else:
                raise ValueError(
                    f'Unexpected pixel count length {n} (expected 256 or 384)'
                )

    duration = integrations * u.s
    counts_qty = u.Quantity(counts_raw, u.ct)
    # counts_comp_err = 0 → pure Poisson errors; compression error is not in the JSON
    # and does not affect visibility values (only amplitude uncertainties).
    zeros_qty = u.Quantity(np.zeros_like(counts_raw), u.ct)

    # pixel_masks / detector_masks: stixpy only checks they are constant across the
    # selected time range, so a single replicated scalar satisfies the check.
    pixel_masks_val = int(sum(l1_json.get('pixel_mask', [0xFF])))
    detector_masks_val = int(l1_json.get('detector_mask', 0xFFFFFFFF))

    data = QTable({
        'counts': counts_qty,
        'counts_comp_err': zeros_qty,
        'triggers': triggers_arr,
        'timedel': duration,
        'rcr': rcr_arr,
        'pixel_masks': np.full(n_t, pixel_masks_val, dtype=int),
        'detector_masks': np.full(n_t, detector_masks_val, dtype=int),
    })

    times = Time(times_unix, format='unix')
    t_starts = times - duration / 2
    t_ends = times + duration / 2
    time_range = TimeRange(t_starts[0], t_ends[-1])

    return SimpleNamespace(
        times=times,
        duration=duration,
        data=data,
        energies=energies,
        energy_masks=SimpleNamespace(energy_mask=energy_mask),
        time_range=time_range,
    )
