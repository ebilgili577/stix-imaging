from __future__ import annotations

import astropy.units as u
from stixpy.coordinates.transforms import get_hpc_info
from sunpy.coordinates import HeliographicStonyhurst

from .coords import get_hpc_coords, get_sun_radius, hpc_to_stix
from .extract import extract_counts, extract_visibilities
from .imaging import (
    calc_chi_score,
    calibrate_visibilities,
    predict_image,
    rotate_image,
)
from .location import predict_location
from .schemas import Selection


def run_imaging_pipeline(
    l1_json: dict,
    selection: Selection,
    mlp_model,
    fcd_model,
    user_hpc_x=None,
    user_hpc_y=None,
):
    result = {}

    raw_counts = extract_counts(l1_json, selection)

    pred_location = predict_location(raw_counts, mlp_model)
    vis, t_center = extract_visibilities(l1_json, selection)
    phase_loc_stix = pred_location  ## use mlp location for imaging
    roll, solo_xyz, _ = get_hpc_info(t_center, t_center)
    observer = HeliographicStonyhurst(
        *solo_xyz,
        obstime=t_center,
        representation_type="cartesian",
    )
    # convert mlp stix to hpc
    mlp_hpc = get_hpc_coords(
        pred_location,
        t_center,
        observer,
    )

    # if user provides coords use those for imaging, else use mlp loc
    if user_hpc_x is not None and user_hpc_y is not None:
        phase_loc_stix = hpc_to_stix(
            float(user_hpc_x), float(user_hpc_y), t_center, observer
        )
        
    cal_vis, phase_loc_hpc = calibrate_visibilities(
        vis, phase_loc_stix, t_center, observer
    )
    img = predict_image(cal_vis, fcd_model)
    rotated_image = rotate_image(img, phase_loc_hpc, roll)

    result["image"] = img
    result["rotated_image"] = rotated_image
    result["sun_radius"] = get_sun_radius(observer)
    result["chi_score"] = calc_chi_score(cal_vis, img)
    result["mlp_stix_x"] = float(pred_location["location_x_arcsec"])
    result["mlp_stix_y"] = float(pred_location["location_y_arcsec"])
    result["mlp_hpc_x"] = float(mlp_hpc.Tx.to_value(u.arcsec))
    result["mlp_hpc_y"] = float(mlp_hpc.Ty.to_value(u.arcsec))
    result["img_hpc_x"] = float(phase_loc_hpc.Tx.to_value(u.arcsec))
    result["img_hpc_y"] = float(phase_loc_hpc.Ty.to_value(u.arcsec))

    return result
