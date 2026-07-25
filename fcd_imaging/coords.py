"""
Module for coords stuff
"""

import astropy.units as u
import sunpy.sun.constants as sun_const
from astropy.coordinates import SkyCoord
from stixpy.coordinates.frames import STIXImaging
from sunpy.coordinates import Helioprojective


def hpc_to_stix(hpc_x: float, hpc_y: float, t_center, observer) -> dict:
    """Convert HPC arcsec to STIX imaging arcsec at ``t_center``."""

    hpc = SkyCoord(
        hpc_x * u.arcsec,
        hpc_y * u.arcsec,
        frame=Helioprojective(observer=observer, obstime=t_center),
    )
    stix = hpc.transform_to(STIXImaging(obstime=t_center))
    return {
        "location_x_arcsec": float(stix.Tx.to_value(u.arcsec)),
        "location_y_arcsec": float(stix.Ty.to_value(u.arcsec)),
    }


def get_hpc_coords(flare_loc: SkyCoord | dict, t_center, observer):
    """Transform a STIXImaging SkyCoord to Helioprojective at ``t_center``."""
    if not isinstance(flare_loc, SkyCoord):
        stix_x, stix_y = flare_loc["location_x_arcsec"], flare_loc["location_y_arcsec"]
        flare_loc = SkyCoord(
            float(stix_x) * u.arcsec,
            float(stix_y) * u.arcsec,
            frame=STIXImaging(obstime=t_center)
        )

    flare_hpc = flare_loc.transform_to(
        Helioprojective(observer=observer, obstime=t_center)
    )
    return flare_hpc


def get_sun_radius(observer):
    """Apparent solar radius [arcsec] as seen from Solar Orbiter`."""
    rsun = (
        (
            sun_const.radius / (observer.spherical.distance - sun_const.radius)
        ).decompose()
        * u.radian
    ).to_value(u.arcsec)
    return float(rsun)
