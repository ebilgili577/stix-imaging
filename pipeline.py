from __future__ import annotations

from coords_util import get_sun_radius, hpc_to_stix
from imaging import calibrate_visibilities, predict_image, rotate_image
from location import predict_location
from stixpy.coordinates.transforms import get_hpc_info
from sunpy.coordinates import HeliographicStonyhurst

from aux_functions import calc_chi_score
from extract import extract_counts, extract_visibilities
from schemas import Selection


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
    # if user provides coords use those for imaging, else use mlp loc
    if user_hpc_x is not None and user_hpc_y is not None:
        phase_loc_stix = hpc_to_stix(
            float(user_hpc_x), float(user_hpc_y), t_center, observer
        )
    cal_vis, phase_loc_hpc = calibrate_visibilities(
        vis, phase_loc_stix, t_center, observer
    )
    flat_image = predict_image(cal_vis, fcd_model)
    rotated_image = rotate_image(flat_image, phase_loc_hpc, roll)
    # TODO: return, predicted stix_loc, predicted hpc, user hpc empty or something, rotated image, flat image, sun_r, chi square, selection, energy channels, and kev

    result["image"] = flat_image
    result["rotated_image"] = rotated_image
    result["sun_radius"] = get_sun_radius(observer)
    result["chi_score"] = calc_chi_score(cal_vis, flat_image)

    return result
