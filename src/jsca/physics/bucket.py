"""Bucket-hydrology soil-moisture reservoir stepping — Isca's Manabe bucket.

Isca's ``idealized_moist_phys.F90`` carries a single-layer soil-moisture
reservoir over land (the classic Manabe 1969 "bucket"): ``bucket_depth`` [m of
water] is filled by precipitation (large-scale condensation + convection) and
drained by surface evaporation, with any excess above the land capacity
(``max_bucket_depth_land``) lost as runoff. Evaporation off an empty bucket is
suppressed and β-ramped in :func:`jsca.physics.surface_flux.surface_flux`; here
we advance the reservoir depth in time.

**The stepping (F90 ``idealized_moist_phys`` L1401-1428).** The reservoir is
marched with the *same* grid-space leapfrog + RAW filter Isca uses for the
column tracers, with the bucket's own Robert/RAW coefficients
(``robert_bucket=0.04``, ``raw_bucket=0.53``)::

    dt_bucket = depth_change_cond + depth_change_conv - depth_change_lh
    filt      = bucket(prev) - 2*bucket(cur)
    ... leapfrog(bucket, dt_bucket) ...            ! future & filtered-current
    where (bucket <= 0) bucket = 0                 ! reservoir cannot go negative
    where (land .and. bucket(fut) > max) bucket(fut) = max   ! runoff

The three ``depth_change_*`` are already *depths per step* [m]:
``depth_change_lh`` comes straight from
:attr:`jsca.physics.surface_flux.SurfaceFluxResult.depth_change_lh`, and the two
precipitation terms are the condensation/convection rain converted to a water
depth by the caller. Because ``dt_bucket`` is the full per-step increment, the
leapfrog is called with ``delta_t = 1.0`` (Isca writes
``bucket(fut) = bucket(prev) + dt_bucket`` with no ``2*dt`` factor — the factor
is folded into the ``depth_change_*`` terms upstream).

**Reuse, not re-derivation.** The L1401-1428 arithmetic is *byte-for-byte* the
combined leapfrog+RAW filter already ported as
:func:`jsca.dycore.leapfrog.leapfrog` (both the ``previous == current`` start-up
branch and the normal branch match line-by-line). So the stepping here is that
function plus the two bucket-specific clamps; the fixture
(``tests/test_bucket_fixtures.py``, from the verbatim Fortran body) validates the
whole composition — leapfrog coefficients, clamp, and runoff order.

Layout: ``bucket_depth`` is ``(..., 2)`` with the leapfrog **time level last**
(mirroring the column model's ``tg``/``qg``); ``land`` is the matching horizontal
mask ``(...)``. The ``<= 0`` clamp is applied to *every* time level, exactly as
Isca's ``where(bucket_depth <= 0.)`` spans the whole array; the runoff clamp
touches only the ``future`` level.
"""
from __future__ import annotations

import jax.numpy as jnp

from jsca.dycore.leapfrog import leapfrog

Array = "jax.Array"

# idealized_moist_phys_nml defaults for the bucket leapfrog filter.
ROBERT_BUCKET = 0.04
RAW_BUCKET = 0.53


def bucket_step(
    bucket_depth: Array,
    depth_change_cond: Array,
    depth_change_conv: Array,
    depth_change_lh: Array,
    land: Array,
    previous: int,
    current: int,
    future: int,
    max_bucket_depth_land: float,
    robert_coeff: float = ROBERT_BUCKET,
    raw_filter_coeff: float = RAW_BUCKET,
) -> Array:
    """Advance the soil-moisture bucket one step (Isca L1401-1428).

    Args:
      bucket_depth: reservoir depth ``(..., 2)`` [m], leapfrog time level last.
      depth_change_cond, depth_change_conv: precipitation depth added this step
        [m] from large-scale condensation and convection (``>= 0``).
      depth_change_lh: evaporative depth removed this step [m] (``>= 0``), i.e.
        :attr:`jsca.physics.surface_flux.SurfaceFluxResult.depth_change_lh`.
      land: horizontal land mask ``(...)`` (bool; runoff only over land).
      previous, current, future: static leapfrog time-level indices into the
        last axis (``previous == current`` selects the Euler-ish first step).
      max_bucket_depth_land: land reservoir capacity [m]; excess runs off.
      robert_coeff, raw_filter_coeff: bucket filter coefficients (Isca
        ``robert_bucket``/``raw_bucket``).

    Returns:
      The updated ``bucket_depth`` array (same shape); the new reservoir depth
      lives in the ``future`` level.
    """
    # dt_bucket is the full per-step depth increment, so delta_t = 1.0.
    dt_bucket = depth_change_cond + depth_change_conv - depth_change_lh
    bd = leapfrog(
        bucket_depth, dt_bucket, previous, current, future,
        1.0, robert_coeff, raw_filter_coeff,
    )
    # Reservoir cannot go negative: Isca clamps every time level.
    bd = jnp.maximum(bd, 0.0)
    # Runoff: excess above capacity is lost, over land only, future level only.
    fut = jnp.where(
        land & (bd[..., future] > max_bucket_depth_land),
        max_bucket_depth_land, bd[..., future],
    )
    return bd.at[..., future].set(fut)
