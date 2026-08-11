"""Idealized-continents land mask — port of Isca's ``land_generator_fn.py``.

Isca prescribes land for the bucket / land configurations by writing a
``land.nc`` with a 0/1 ``land_mask`` on the model grid; the model then reads it
(``land_option='input'``) and every land-aware routine — the mixed-layer heat
capacity/albedo (:func:`jsca.physics.mixed_layer.land_heat_capacity` /
:func:`jsca.physics.mixed_layer.land_albedo`), the bucket evaporation switch
(:mod:`jsca.physics.surface_flux`) and the bucket water budget
(:mod:`jsca.physics.bucket`) — keys off that mask.

:func:`continents_land_mask` is the faithful port of the ``land_mode='continents'``
branch of ``src/extra/python/isca/land_generator_fn.py`` (L84-105): the newer
Sauliere-2012-derived continent set (North & South America, Eurasia, Africa,
Australia, India, South East Asia), each a set of half-plane / box inequalities in
degrees of latitude and longitude. Isca hard-codes ``t_res=42`` and reads the grid
from a file; here the grid latitudes/longitudes are passed in, so the *same*
geometry is evaluated at any resolution (T21 for the fast bucket validation, T42 to
reproduce Isca's own ``write_land`` output byte-for-byte — see
``tests/test_land_mask.py``).

**Note on the shipped bucket test case.** Isca's
``exp/test_cases/bucket_hydrology/input/land.nc`` is actually a *square* land block
(``land_mode='square'``), not these continents. This module implements the
realistic-continents geometry; the jsca bucket model and its Isca counterpart are
both driven from the continents mask generated here, so the comparison is
like-for-like.

Longitude convention matches Isca: ``0 <= lon < 360`` increasing eastward, latitude
south → north (:mod:`jsca.grid`). Returns a ``float64`` ``(nlat, nlon)`` array of
0.0 / 1.0, exactly as ``land.nc`` stores it; take ``> 0.5`` for the boolean mask.
"""
from __future__ import annotations

import numpy as np

# Continent-name -> row in the idx_c stack (land_generator_fn.py L82).
_CONT_INDEX = {"NA": 0, "SA": 1, "EA": 2, "AF": 3, "OZ": 4, "IN": 5, "SEA": 6}


def continents_land_mask(lat_deg, lon_deg, continents=("all",)) -> np.ndarray:
    """Idealized-continents land mask on a lat/lon grid (Isca ``land_generator_fn``).

    Args:
      lat_deg: 1-D latitudes in degrees, south → north, length ``nlat``.
      lon_deg: 1-D longitudes in degrees, ``0 <= lon < 360``, length ``nlon``.
      continents: ``("all",)`` for every continent (default), or an iterable of
        the codes ``NA SA EA AF OZ IN SEA`` to include a subset.

    Returns:
      ``(nlat, nlon)`` float64 array, 1.0 over land and 0.0 over ocean.
    """
    lon_array, lat_array = np.meshgrid(np.asarray(lon_deg, dtype=np.float64),
                                       np.asarray(lat_deg, dtype=np.float64))
    nlat, nlon = lat_array.shape
    idx_c = np.zeros((7, nlat, nlon), dtype=bool)

    # North America (land_generator_fn.py L86)
    idx_c[0] = ((103. - 43. / 40. * (lon_array - 180) < lat_array)
                & ((lon_array - 180) * 43. / 50. - 51.8 < lat_array)
                & (lat_array < 60.))
    # South America (L87)
    idx_c[1] = ((737. - 7.2 * (lon_array - 180) < lat_array)
                & ((lon_array - 180) * 10. / 7. + -212.1 < lat_array)
                & (lat_array < -22. / 45 * (lon_array - 180) + 65.9))
    # Eurasia — split across the 0/360 seam (L88-90)
    eurasia_pos = ((23. <= lat_array) & (lat_array < 60.) & (-8. < lon_array)
                   & (43. / 40. * lon_array - 101.25 < lat_array))
    eurasia_neg = (23. <= lat_array) & (lat_array < 60.) & (352. < lon_array)
    idx_c[2] = eurasia_pos | eurasia_neg
    # Africa — split across the 0/360 seam (L91-93)
    africa_pos = ((lat_array < 23.) & (-52. / 27. * lon_array + 7.59 < lat_array)
                  & (52. / 38. * lon_array - 65.1 < lat_array))
    africa_neg = (lat_array < 23.) & (-52. / 27. * (lon_array - 360) + 7.59 < lat_array)
    idx_c[3] = africa_pos | africa_neg
    # Australia (L94)
    idx_c[4] = ((lat_array > -35.) & (lat_array < -17.)
                & (lon_array > 115.) & (lon_array < 150.))
    # India (L95)
    idx_c[5] = ((lat_array < 23.) & (-15. / 8. * lon_array + 152 < lat_array)
                & (15. / 13. * lon_array - 81 < lat_array))
    # South East Asia (L96)
    idx_c[6] = ((lat_array < 23.) & (43. / 40. * lon_array - 101.25 < lat_array)
                & (-14. / 13. * lon_array + 120 < lat_array))

    if "all" in continents:
        idx = idx_c.any(axis=0)
    else:
        idx = np.zeros((nlat, nlon), dtype=bool)
        for cont in continents:
            idx |= idx_c[_CONT_INDEX[cont]]

    land = np.zeros((nlat, nlon), dtype=np.float64)
    land[idx] = 1.0
    return land
