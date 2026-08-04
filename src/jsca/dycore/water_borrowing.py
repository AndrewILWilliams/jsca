"""Negative-humidity "hole filling" by borrowing from neighbouring cells.

Faithful port of Isca's ``src/atmos_spectral/model/water_borrowing.F90``. Where a
grid cell has negative specific humidity, it borrows water from its horizontal
(east/west, periodic in longitude) and vertical (level above/below) neighbours,
scaling each neighbour down by a common ``ratio`` so the local + neighbour water
is redistributed to remove the hole while conserving total water in the
5-point stencil. The correction is returned as a tendency increment on
``dt_qg`` (units 1/s), exactly as the Fortran.

Layout: grid fields are ``(..., nlat, nlon, K)`` with longitude the periodic
axis (``-2``) and level last (``-1``); ``p_half`` is ``(..., nlat, nlon, K+1)``.
The Fortran storage is ``(lon, lat, level)``, so fixtures transpose lon<->lat.

Faithful-but-vectorised: the Fortran sweeps longitude sequentially (direction set
by ``current``'s parity) and scatter-accumulates into ``dt_qg``, but it reads
**only the original ``qg``** (never the running ``dt_qg``) — so the mathematical
result is independent of sweep direction; the parity only changes the
floating-point accumulation order. This port computes the identical
scatter-accumulate in closed form (each hole's ``ratio`` from the original
``qg``), so ``current`` affects nothing here and is omitted. For a cell adjacent
to more than one hole the summation order of the neighbour contributions may
differ from the Fortran sweep at the last bit; agreement is otherwise exact.

The commented-out alternative branch in the Fortran (``total_water < 0``) is
inactive there and is not ported.
"""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def _level_down(x: Array) -> Array:
    """``result[..., k] = x[..., k-1]``; 0 at the top level (no ``k-1`` neighbour)."""
    return jnp.concatenate([jnp.zeros_like(x[..., :1]), x[..., :-1]], axis=-1)


def _level_up(x: Array) -> Array:
    """``result[..., k] = x[..., k+1]``; 0 at the surface level (no ``k+1`` neighbour)."""
    return jnp.concatenate([x[..., 1:], jnp.zeros_like(x[..., :1])], axis=-1)


def water_borrowing(dt_qg: Array, qg: Array, p_half: Array, delta_t: float) -> Array:
    """Add the hole-filling tendency to ``dt_qg`` — port of ``water_borrowing``.

    ``qg``/``dt_qg`` are ``(..., nlat, nlon, K)``; ``p_half`` is ``(..., nlat, nlon, K+1)``.
    Longitude (axis ``-2``) is periodic; levels (axis ``-1``) are not.
    Returns the updated ``dt_qg``.
    """
    dp = p_half[..., 1:] - p_half[..., :-1]
    water = qg * dp  # qg * dp per cell

    # neighbour water: east/west (periodic lon) + above/below (level, zero-padded)
    neighbouring_water = (
        jnp.roll(water, 1, axis=-2)
        + jnp.roll(water, -1, axis=-2)
        + _level_down(water)
        + _level_up(water)
    )
    total_water = neighbouring_water + water

    is_hole = (qg < 0.0) & (total_water > 0.0)
    # ratio only used where is_hole; guard the divide elsewhere (in-branch the
    # Fortran always has neighbouring_water > 0 when total_water > 0 and qg < 0).
    ratio = total_water / jnp.where(neighbouring_water != 0.0, neighbouring_water, 1.0)
    hole_factor = jnp.where(is_hole, ratio - 1.0, 0.0)

    # each hole removes its own water; each of its neighbours c contributes
    # (ratio_c - 1) * qg[x]/dt to cell x — i.e. qg[x]/dt times the hole_factor at
    # x's neighbours (mirror of the neighbour stencil above).
    q_over_dt = qg / delta_t
    self_term = jnp.where(is_hole, -q_over_dt, 0.0)
    neighbour_term = q_over_dt * (
        jnp.roll(hole_factor, 1, axis=-2)
        + jnp.roll(hole_factor, -1, axis=-2)
        + _level_down(hole_factor)
        + _level_up(hole_factor)
    )
    return dt_qg + self_term + neighbour_term
