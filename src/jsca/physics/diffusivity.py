"""Boundary-layer eddy diffusivity — Isca's ``diffusivity.F90`` (simple K-profile).

Faithful port of the non-local K-profile scheme ``vert_turb_driver`` calls when
``do_diffusivity = .true.``, for the Frierson aquaplanet settings
(``diffusivity_nml: do_simple=.true., do_entrain=.false.``; defaults
``fixed_depth=F, free_atm_diff=F, pbl_mcm=F, background_m/t=0, frac_inner=0.1,
rich_crit_pbl=1.0, use_pog_bug_fix=T``). It returns the eddy diffusivities for
momentum (``k_m``) and heat/tracers (``k_t``) on the half levels, plus the
planetary-boundary-layer (PBL) depth ``h``.

With ``constant_gust = 0`` the surrounding ``vert_turb_driver`` reduces, for
Frierson, to this call plus ``gust = 0`` and ``z_pbl = h`` — so this module is the
substance of item 6.

**Algorithm (F90 ``diffusivity`` + ``pbl_depth`` + ``diffusivity_pbl``):**

1. Dry static energy over cp (do_simple): ``svcp = t + (g/cp)*z``, heights ``z``
   measured above the surface (F90 L308).
2. **PBL depth** (do_simple → the neutral/stable Richardson branch, F90 L412):
   a bulk Richardson number ``Ri(k) = z*g*(svcp - svcp_sfc)/svcp_sfc /
   (u^2+v^2)`` is walked upward from the surface until it exceeds
   ``rich_crit_pbl``; ``h`` is the linearly-interpolated crossing height.
3. **K profile** (F90 ``diffusivity_pbl``): with ``h_inner = frac_inner*h``, the
   Monin-Obukhov surface-layer diffusivity (``mo_diff``) is used below
   ``h_inner``; between ``h_inner`` and ``h`` the profile is the cubic
   ``K = K_ref * (z/h_inner) * (1 - (z-h_inner)/(h-h_inner))^2`` with ``K_ref``
   the surface-layer value at ``h_inner``; above ``h`` it is zero
   (``use_pog_bug_fix``).

**Scope:** the Frierson simple path. Entrainment, free-atmosphere diffusion, the
MCM PBL, the LCL-depth option, molecular diffusion, and non-zero background
diffusivities are not ported (flagged here). No documented deviation — pure
arithmetic (the ``mo_diff`` similarity functions are ``log``/powers), so fixtures
hit the log/exp tolerance.

Layout: column physics, **level axis last**; ``t``/``u``/``v``/``z_full`` are
``(..., K)``, ``z_half`` is ``(..., K+1)`` (``z_half[..., K]`` the surface),
``u_star``/``b_star`` are ``(...)``. ``k_m``/``k_t`` are returned ``(..., K)`` at
the first K half levels (index ``k`` = the upper interface of full level ``k``;
``k=0`` the model top → 0, the surface half level excluded).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jsca import constants
from jsca.physics.monin_obukhov import MOParams, mo_diff

Array = jnp.ndarray


@dataclass(frozen=True)
class DiffusivityParams:
    """Static ``diffusivity_nml`` configuration (hashable; static jit arg)."""

    frac_inner: float = 0.1
    rich_crit_pbl: float = 1.0
    small: float = 1.0e-4
    background_m: float = 0.0
    background_t: float = 0.0


def _pbl_depth(dp: DiffusivityParams, svcp, u, v, z, u_star, b_star):
    """PBL depth from the bulk Richardson number (F90 ``pbl_depth``, do_simple).

    ``svcp``/``u``/``v``/``z`` are ``(..., K)`` (heights above surface); returns
    ``h`` ``(...)``. Walks up from the surface to the first ``Ri > rich_crit_pbl``
    crossing and interpolates.
    """
    tbot = svcp[..., -1:]                       # surface full level
    rich = z * constants.GRAV * (svcp - tbot) / tbot / (u * u + v * v + dp.small)

    # reorder surface -> top so the scan walks upward
    rich_r = jnp.moveaxis(rich[..., ::-1], -1, 0)   # (K, ...), index 0 = surface
    z_r = jnp.moveaxis(z[..., ::-1], -1, 0)

    def body(carry, level):
        h, rich1, h1, done = carry
        rich2, h2 = level
        cross = (~done) & (rich2 > dp.rich_crit_pbl)
        denom = jnp.where(rich2 - rich1 != 0.0, rich2 - rich1, 1.0)
        h_new = h2 + (h1 - h2) * (rich2 - dp.rich_crit_pbl) / denom
        h = jnp.where(cross, h_new, h)
        advance = (~done) & (~cross)
        rich1 = jnp.where(advance, rich2, rich1)
        h1 = jnp.where(advance, h2, h1)
        done = done | cross
        return (h, rich1, h1, done), None

    h0 = z_r[0]                                  # surface height (above surface)
    init = (h0, rich_r[0], h0, jnp.zeros(h0.shape, bool))
    (h, _, _, _), _ = jax.lax.scan(body, init, (rich_r[1:], z_r[1:]))
    return h


def diffusivity(dp: DiffusivityParams, t, q, u, v, z_full, z_half,
                u_star, b_star, mo_params: MOParams = MOParams()):
    """Simple K-profile diffusivity. Returns ``(k_m, k_t, h)``.

    ``t``/``u``/``v``/``z_full`` are ``(..., K)``; ``z_half`` is ``(..., K+1)``;
    ``u_star``/``b_star`` are ``(...)``. ``q`` is accepted for signature
    compatibility but unused on the do_simple path.
    """
    gcp = constants.GRAV / constants.CP_AIR
    z_surf = z_half[..., -1:]                    # surface half level
    z_full_ag = z_full - z_surf
    z_half_ag = z_half - z_surf

    svcp = t + gcp * z_full_ag                   # dry static energy / cp (do_simple)
    h = _pbl_depth(dp, svcp, u, v, z_full_ag, u_star, b_star)

    # --- K profile (diffusivity_pbl) ---
    h_inner = dp.frac_inner * h                  # (...,)
    zm = z_half_ag[..., :-1]                      # first K half levels (..., K)
    us = u_star[..., None]
    bs = b_star[..., None]

    k_m_ref, k_t_ref = mo_diff(mo_params, h_inner, u_star, b_star)   # (...,)
    k_m_full, k_t_full = mo_diff(mo_params, zm, us, bs)              # (..., K)

    hi = h_inner[..., None]
    hh = h[..., None]
    denom = jnp.where(hh - hi != 0.0, hh - hi, 1.0)
    factor = (zm / hi) * (1.0 - (zm - hi) / denom) ** 2

    inner = zm < hi
    outer = (~inner) & (zm < hh)
    k_m = jnp.where(inner, k_m_full, jnp.where(outer, k_m_ref[..., None] * factor, 0.0))
    k_t = jnp.where(inner, k_t_full, jnp.where(outer, k_t_ref[..., None] * factor, 0.0))

    if dp.background_m > 0.0:
        k_m = jnp.maximum(k_m, dp.background_m)
    if dp.background_t > 0.0:
        k_t = jnp.maximum(k_t, dp.background_t)
    return k_m, k_t, h
