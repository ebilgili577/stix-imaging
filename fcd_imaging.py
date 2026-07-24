"""Module for imaging related functions.
- Visibilities and calibration
- Image prediction
- Image rotation
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from coords_util import get_hpc_coords
from stixpy.calibration.visibility import calibrate_visibility
from stixpy.coordinates.frames import STIXImaging
from sunpy.map import Map, make_fitswcs_header

from aux_functions import Fourier_matrix_STIX, compute_chi2

# FCD was trained on 24 visibilities (rings 3–10, a/b/c each), not stixpy's full 30.
# Label order matches fcd/integration_utils.py and the STIX L3A .sav training format.
FCD_VIS_LABELS: tuple[str, ...] = tuple(
    f"{ring}{suffix}" for ring in range(10, 2, -1) for suffix in "abc"
)


def predict_image(cal_vis, fcd) -> list:
    """Run FCD on a (48,) Re/Im vector; return flattened 128×128 list."""
    if fcd is None:
        raise RuntimeError("FCD model not loaded")
    fcd_input = visibilities_to_fcd_input(cal_vis)
    predicted = np.squeeze(fcd.predict(fcd_input.reshape(1, -1), verbose=0))
    return predicted.tolist()


def rotate_image(flat_image: list[float], hpc_coord: SkyCoord, roll):
    """Build a sunpy Map in HPC, rotate to north-up, return image + axis coords.

    FCD output is a 128×128 STIX-oriented array at 2″/pix. We attach an HPC
    WCS centered on ``hpc_coord`` with rotation_angle = 90° + Solo roll, then
    ``Map.rotate()`` produces a north-up array. Axis vectors ``x``/``y`` are
    world Tx/Ty [arcsec] along the mid-row / mid-column for Plotly.
    """
    img = np.array(flat_image).reshape(128, 128)

    header_hp = make_fitswcs_header(
        img,
        hpc_coord,
        scale=[2, 2] * u.arcsec / u.pix,
        rotation_angle=90 * u.deg + roll,
    )
    hp_map = Map((img, header_hp))
    hp_map_rotated = hp_map.rotate()

    # fill nan with 0z for json serialization
    data = np.nan_to_num(np.asarray(hp_map_rotated.data, dtype=np.float64))
    ny, nx = data.shape
    px = np.arange(nx) * u.pix
    py = np.arange(ny) * u.pix
    # Sample world coords along the image mid-axes (Plotly heatmap x/y).
    world_x = hp_map_rotated.pixel_to_world(px, np.full(nx, ny / 2) * u.pix)
    world_y = hp_map_rotated.pixel_to_world(np.full(ny, nx / 2) * u.pix, py)
    x = world_x.Tx.to_value(u.arcsec).tolist()
    y = world_y.Ty.to_value(u.arcsec).tolist()
    return {
        "image": data.tolist(),
        "x": x,
        "y": y,
    }


def calibrate_visibilities(vis, location: dict, t_center, observer):
    """calibrates visibilities for fcd input and returns the locations used for calibrating in hpc frame."""
    flare_loc = SkyCoord(
        location["location_x_arcsec"] * u.arcsec,
        location["location_y_arcsec"] * u.arcsec,
        frame=STIXImaging(obstime=t_center),
    )
    return calibrate_visibility(vis, flare_loc), get_hpc_coords(
        flare_loc, t_center, observer
    )


def visibilities_to_fcd_input(cal_vis) -> np.ndarray:
    """Convert stixpy Visibilities to the 48-dim vector the FCD model expects.

    stixpy returns 30 imaging visibilities (rings 1–10). FCD uses only the 24
    coarsest rings (3–10) in label order 10a…10c, 9a…9c, …, 3a…3c, matching
    the STIX L3A .sav format used during FCD training.
    Layout: [Re(24), Im(24)].
    """
    labels = [str(label) for label in cal_vis.meta["vis_labels"]]
    by_label = {label: value for label, value in zip(labels, cal_vis.visibilities)}
    missing = [label for label in FCD_VIS_LABELS if label not in by_label]
    if missing:
        raise ValueError(f"Missing FCD visibility labels: {missing}")
    v = np.asarray([by_label[label] for label in FCD_VIS_LABELS])
    return np.hstack((np.real(v), np.imag(v))).astype(np.float32)


def calc_chi_score(cal_vis, image):
    """Reduced χ² between FCD image forward-vis and calibrated amplitudes.

    Builds the STIX Fourier matrix for the FCD (u,v) sampling, projects the
    128×128 image to complex visibilities, and compares to the observed
    Re/Im vector (``visibilities_to_fcd_input``) with amplitude uncertainties.
    """
    mem_im = np.array(image).reshape(128, 128)

    labels = [str(label) for label in cal_vis.meta["vis_labels"]]
    idx = [labels.index(lab) for lab in FCD_VIS_LABELS]

    vis = visibilities_to_fcd_input(cal_vis)  # 48-dim: Re then Im
    uu = cal_vis.u[idx].to(1 / u.arcsec).value
    vv = cal_vis.v[idx].to(1 / u.arcsec).value
    sigamp = np.asarray(cal_vis.amplitude_uncertainty[idx].to_value(), dtype=np.float64)
    print(sigamp)
    # TODO: check order
    # make plot of amp of visibilities, take vis as input, plot amp of those,
    # compare amp visibilities of reconstructed image
    # send example to paolo with example, raw image,

    n_pix = 128  # FCD output size
    pix_size = 2.0  # arcsec / pixel

    F = Fourier_matrix_STIX(uu, vv, n_pix, pix_size)

    dim = mem_im.shape
    vis_mem_ge = F @ np.reshape(mem_im, (dim[0] * dim[1]))

    chi2 = float(compute_chi2(vis, vis_mem_ge, sigamp))
    print(chi2)
    return round(chi2, 2)
