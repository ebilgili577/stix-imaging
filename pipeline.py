"""Count extraction, MLP location prediction, and visibility computation.

The MLP path (extract_counts → predict_location) is a lightweight numpy-only
routine that avoids all stixpy overhead.

The visibility path (extract_visibilities → calibrate_visibilities) builds a
duck-typed pixel_data adapter (l1_pixel_data.py) and delegates every physics
step — livetime correction, ELUT edge correction, rate/keV/cm² normalisation,
moire-pattern phase, grid/phase calibration — to unmodified stixpy functions.
"""
from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.time import TimeRange
from sunpy.coordinates import  Helioprojective

from schemas import Selection, UserHpc
from l1_pixel_data import pixel_data_from_l1_json
from stixpy.calibration.energy import get_sci_channels
from stixpy.calibration.visibility import (
    create_meta_pixels,
    create_visibility,
    calibrate_visibility,
)
from stixpy.coordinates.frames import STIXImaging
from stixpy.coordinates.transforms import get_hpc_info
from sunpy.coordinates import HeliographicStonyhurst

from sunpy.coordinates.sun import angular_radius

from aux_functions import Fourier_matrix_STIX, compute_chi2

from sunpy.map import Map, make_fitswcs_header


# TODO: clean up or move away
NORMALIZATION_FACTOR = 4000.0

# 1-based detector labels in MLP training order (stix_train DETECTOR_ORDER).
DETECTOR_ORDER = np.array(
    [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28, 15, 27, 31, 6, 30, 2, 25, 5, 23, 7, 29, 1],
    dtype=int,
)

# FCD was trained on 24 visibilities (rings 3–10, a/b/c each), not stixpy's full 30.
# Label order matches fcd/integration_utils.py and the STIX L3A .sav training format.
FCD_VIS_LABELS: tuple[str, ...] = tuple(
    f'{ring}{suffix}' for ring in range(10, 2, -1) for suffix in 'abc'
)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_imaging_pipeline(
    l1_json: dict,
    selection: Selection,
    mlp_model,
    fcd_model=None,
    user_hpc: UserHpc | None = None,
):
    result = {}
    raw_counts = extract_counts(l1_json, selection)
    if mlp_model is not None:
        location = predict_location(raw_counts, mlp_model)
        vis = extract_visibilities(l1_json, selection)
        t_center = TimeRange(vis.meta.time_range).center

        if user_hpc is not None:
            phase_location = hpc_to_stix(user_hpc.hpc_x, user_hpc.hpc_y, t_center)
            result['user_hpc_x'] = float(user_hpc.hpc_x)
            result['user_hpc_y'] = float(user_hpc.hpc_y)
            result['location_source'] = 'user'
        else:
            phase_location = location
            result['location_source'] = 'mlp'

        cal_vis, flare_loc, t_center = calibrate_visibilities(vis, phase_location)
        result.update(location)

        mlp_flare_loc = SkyCoord(
            location['location_x_arcsec'] * u.arcsec,
            location['location_y_arcsec'] * u.arcsec,
            frame=STIXImaging(obstime=t_center),
        )
        mlp_hpc = get_hpc_coords(mlp_flare_loc, t_center)
        result['mlp_hpc_x'] = float(mlp_hpc.Tx.to_value(u.arcsec))
        result['mlp_hpc_y'] = float(mlp_hpc.Ty.to_value(u.arcsec))

    if fcd_model is not None:
        fcd_input = visibilities_to_fcd_input(cal_vis)
        flat_image = predict_image(fcd_input, fcd_model)
        result['image'] = flat_image
   

        hpc_coord, sun_radius, chi_score = get_meta(flare_loc, t_center, flat_image, cal_vis)
        rotated_image = rotate_image(flat_image, hpc_coord, t_center)
        result['rotated_image'] = rotated_image

        result['hpc_x'] = float(hpc_coord.Tx.to_value(u.arcsec))
        result['hpc_y'] = float(hpc_coord.Ty.to_value(u.arcsec))
        result['sun_radius'] = sun_radius
        result['chi_score'] = chi_score

    return result


# TODO: cache this info for get hpc info, or dont call it somehow, we need roll tho
def rotate_image(flat_image: list[float], hpc_coord: SkyCoord, t_center):
    img = np.array(flat_image).reshape(128, 128)
    roll, _, __ = get_hpc_info(t_center, t_center)

    header_hp = make_fitswcs_header(
        img,
        hpc_coord,
        scale=[2, 2] * u.arcsec / u.pix,
        rotation_angle=90 * u.deg + roll,
    )
    hp_map = Map((img, header_hp))
    hp_map_rotated = hp_map.rotate()

    data = np.nan_to_num(
        np.asarray(hp_map_rotated.data, dtype=np.float64)
    )
    ny, nx = data.shape
    px = np.arange(nx) * u.pix
    py = np.arange(ny) * u.pix
    world_x = hp_map_rotated.pixel_to_world(px, np.full(nx, ny / 2) * u.pix)
    world_y = hp_map_rotated.pixel_to_world(np.full(ny, nx / 2) * u.pix, py)
    x = world_x.Tx.to_value(u.arcsec).tolist()
    y = world_y.Ty.to_value(u.arcsec).tolist()
    return {
        "image": data.tolist(),
        "x": x,
        "y": y,
    }


def hpc_to_stix(hpc_x: float, hpc_y: float, t_center) -> dict:
    """Convert user HPC arcsec to STIX imaging arcsec at ``t_center``."""
    _, solo_xyz, _ = get_hpc_info(t_center, t_center)
    solo = HeliographicStonyhurst(
        *solo_xyz, obstime=t_center, representation_type='cartesian',
    )
    hpc = SkyCoord(
        hpc_x * u.arcsec,
        hpc_y * u.arcsec,
        frame=Helioprojective(observer=solo, obstime=t_center),
    )
    stix = hpc.transform_to(STIXImaging(obstime=t_center))
    return {
        'location_x_arcsec': float(stix.Tx.to_value(u.arcsec)),
        'location_y_arcsec': float(stix.Ty.to_value(u.arcsec)),
    }


def get_meta(flare_loc: SkyCoord, t_center, flat_image, cal_vis):
    hpc_coords = get_hpc_coords(flare_loc, t_center)
    sun_radius = get_sun_radius(t_center)
    chi_score = calc_chi_score(cal_vis, flat_image)
    return hpc_coords, sun_radius, chi_score


def get_hpc_coords(flare_loc: SkyCoord, center):
    flare_hpc = flare_loc.transform_to(Helioprojective(obstime=center))
    return flare_hpc


def get_sun_radius(center):
    sun_radius_arcsec = angular_radius(center).value

    return sun_radius_arcsec

def calc_chi_score(cal_vis, image):

    mem_im = np.array(image).reshape(128, 128)
   
    FCD_VIS_LABELS = tuple(
    f'{ring}{suffix}' for ring in range(10, 2, -1) for suffix in 'abc'
)

    labels = [str(label) for label in cal_vis.meta['vis_labels']]
    idx = [labels.index(lab) for lab in FCD_VIS_LABELS]

    vis = visibilities_to_fcd_input(cal_vis)
    uu = cal_vis.u[idx].to(1 / u.arcsec).value
    vv = cal_vis.v[idx].to(1 / u.arcsec).value
    sigamp = np.asarray(cal_vis.amplitude_uncertainty[idx].to_value(), dtype=np.float64)
    print(sigamp)
    # TODO: check order
    # make plot of amp of visibilities, take vis as input, plot amlp of those,
    # compare amplt visiblities of reconstructed image
    # send example to paolo with example, raw image, 

    n_pix = 128     # output_size
    pix_size = 2.0  # pixel size in arcsec

    F = Fourier_matrix_STIX(uu, vv, n_pix, pix_size)      

    dim = mem_im.shape
    vis_mem_ge = F @ np.reshape(mem_im, (dim[0]*dim[1]))

    chi2 = float(compute_chi2(vis, vis_mem_ge, sigamp))
    print(chi2)
    return round(chi2, 2)




# ---------------------------------------------------------------------------
# MLP path — fast, numpy-only count extraction
# ---------------------------------------------------------------------------

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
        f'Unexpected L1 pixel count length {n} (expected 256 or 384)',
    )


def extract_counts(l1_json: dict, selection: Selection) -> np.ndarray:
    counts = np.zeros((32, 8), dtype=np.float32)
    for box in l1_json['boxes']:
        if not box:
            continue
        if not _box_within_time(box, selection.start_unix, selection.end_unix):
            continue
        for energy_channel in box['counts']:
            if not _in_energy_channel(
                energy_channel, selection.e_channel_min, selection.e_channel_max,
            ):
                continue
            counts += _flat_to_big_pixel_counts(
                np.asarray(energy_channel[3], dtype=np.float32),
            )
    # Rows in DETECTOR_ORDER; cols 0–7 = top abcd + bottom abcd.
    return counts[DETECTOR_ORDER - 1]


# ---------------------------------------------------------------------------
# Visibility path — stixpy adapter
# ---------------------------------------------------------------------------

def extract_visibilities(l1_json: dict, selection: Selection):
    """Return uncalibrated stixpy Visibilities for the given selection.

    Builds a duck-typed pixel_data from the L1 JSON and delegates to
    stixpy create_meta_pixels (with no_shadowing=True, pixels='top+bot')
    and create_visibility — both called unmodified.
    """
    pixel_data = pixel_data_from_l1_json(l1_json)
    time_range = Time([selection.start_unix, selection.end_unix], format='unix')
    energy_range = _channels_to_energy_range(pixel_data, selection)
    meta = create_meta_pixels(
        pixel_data, time_range, energy_range, pixels='top+bot', no_shadowing=True,
    )
    return create_visibility(meta)


def calibrate_visibilities(vis, location: dict):
    """Phase-calibrate stixpy Visibilities using the MLP-predicted location.

    Constructs the SkyCoord in STIXImaging with obstime == TimeRange.center
    so that calibrate_visibility skips the get_hpc_info ephemeris download
    (the condition ``isinstance(..., STIXImaging) and obstime == tr.center``).
    """
    tr = TimeRange(vis.meta.time_range)
    flare_loc = SkyCoord(
        location['location_x_arcsec'] * u.arcsec,
        location['location_y_arcsec'] * u.arcsec,
        frame=STIXImaging(obstime=tr.center),
    )
    return calibrate_visibility(vis, flare_loc), flare_loc, tr.center


def visibilities_to_fcd_input(cal_vis) -> np.ndarray:
    """Convert stixpy Visibilities to the 48-dim vector the FCD model expects.

    stixpy returns 30 imaging visibilities (rings 1–10). FCD uses only the 24
    coarsest rings (3–10) in label order 10a…10c, 9a…9c, …, 3a…3c, matching
    the STIX L3A .sav format used during FCD training.
    """
    labels = [str(label) for label in cal_vis.meta['vis_labels']]
    by_label = {label: value for label, value in zip(labels, cal_vis.visibilities)}
    missing = [label for label in FCD_VIS_LABELS if label not in by_label]
    if missing:
        raise ValueError(f'Missing FCD visibility labels: {missing}')
    v = np.asarray([by_label[label] for label in FCD_VIS_LABELS])
    return np.hstack((np.real(v), np.imag(v))).astype(np.float32)


# ---------------------------------------------------------------------------
# ML models
# ---------------------------------------------------------------------------

def predict_location(raw_counts: np.ndarray, mlp) -> dict:
    if mlp is None:
        raise RuntimeError('MLP model not loaded')
    col_count = mlp.input_shape[1] // 8
    X = _counts_to_mlp_features(raw_counts, col_count)
    X = normalize(X)
    preds = mlp.predict(X, verbose=0)
    x_arcsec, y_arcsec = (preds[0] * NORMALIZATION_FACTOR).tolist()
    return {
        'status': 'OK',
        'location_x_arcsec': float(x_arcsec),
        'location_y_arcsec': float(y_arcsec),
    }


def predict_image(fcd_input: np.ndarray, fcd) -> list:
    if fcd is None:
        raise RuntimeError('FCD model not loaded')
    predicted = np.squeeze(fcd.predict(fcd_input.reshape(1, -1), verbose=0))
    return predicted.tolist()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _channels_to_energy_range(pixel_data, selection: Selection):
    """Return [e_low, e_high] Quantity[keV] spanning the selection's channel range.

    Uses the stixpy science-channel table (date-dependent, internally cached)
    to convert integer channel indices to keV boundaries.
    """
    obstime = pixel_data.time_range.center
    sci_ch = get_sci_channels(obstime)
    ch_nums = np.asarray(sci_ch['Channel Number'])

    idx_lo = np.flatnonzero(ch_nums == selection.e_channel_min)
    idx_hi = np.flatnonzero(ch_nums == selection.e_channel_max)
    if not idx_lo.size or not idx_hi.size:
        raise ValueError(
            f'Science channel {selection.e_channel_min} or {selection.e_channel_max} '
            f'not found in channel table for {obstime}'
        )
    e_low = sci_ch['Elower'][idx_lo[0]]
    e_high = sci_ch['Eupper'][idx_hi[0]]
    return u.Quantity([e_low.to_value('keV'), e_high.to_value('keV')], u.keV)


def _box_within_time(box: dict, t_start: float, t_end: float) -> bool:
    half_dur = float(box['integrations']) / 2.0
    tcenter = float(box['time'])
    return tcenter + half_dur >= t_start and tcenter - half_dur <= t_end


def _in_energy_channel(energy_channel, ch_min: int, ch_max: int) -> bool:
    e1 = int(energy_channel[0])
    e2 = int(energy_channel[1])
    return e1 >= ch_min and e2 <= ch_max


def _counts_to_mlp_features(raw_counts: np.ndarray, col_count: int) -> np.ndarray:
    """(col_count, 8) → (1, col_count * 8) in training column order."""
    features = []
    for i in range(col_count):
        r = raw_counts[i]
        # row layout: top_a,b,c,d | bot_a,b,c,d  →  a_top,a_bot,b_top,b_bot,...
        features.extend([r[0], r[4], r[1], r[5], r[2], r[6], r[3], r[7]])
    return np.array(features, dtype=np.float64).reshape(1, -1)


def normalize(X: np.ndarray) -> np.ndarray:
    return X / X.max(axis=1, keepdims=True)

