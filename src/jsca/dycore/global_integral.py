"""Mass-weighted, globally area-averaged vertical integral of a grid field.

Faithful port of Isca's ``src/atmos_spectral/model/global_integral.F90``. Returns
the vertical mass integral of ``field`` (``sum_k field · dp / g``, units
``(field units)·kg/m^2``) area-averaged over the globe — the diagnostic used to
monitor conservation.

Layout: grid fields are ``(..., nlat, nlon, K)`` with level last; ``surf_press``
is ``(..., nlat, nlon)``. The pressure thickness ``dp`` comes from the ported
:func:`jsca.dycore.half_level_pressures`; the horizontal area mean is the ported
:func:`jsca.grid.area_weighted_global_mean` (Gaussian latitude weights).
"""

from __future__ import annotations

import jax.numpy as jnp

from jsca import constants
from jsca.dycore.press_and_geopot import half_level_pressures
from jsca.grid.transforms import TransformParams, area_weighted_global_mean

Array = jnp.ndarray


def mass_weighted_global_integral(
    params: TransformParams,
    pk: Array,
    bk: Array,
    field: Array,
    surf_press: Array,
    grav: float = constants.GRAV,
) -> Array:
    """Port of ``mass_weighted_global_integral`` (global_integral.F90 L49-81).

    ``pk``/``bk`` are the ``K+1`` hybrid coefficients; ``field`` is
    ``(..., nlat, nlon, K)`` and ``surf_press`` ``(..., nlat, nlon)``. Returns the
    scalar (per leading batch) globally area-averaged mass integral.
    """
    p_half = half_level_pressures(pk, bk, surf_press)  # (..., nlat, nlon, K+1)
    dp = p_half[..., 1:] - p_half[..., :-1]
    vert_integral = jnp.sum(field * dp, axis=-1)  # (..., nlat, nlon)
    return area_weighted_global_mean(params, vert_integral) / grav
