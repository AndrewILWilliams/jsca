"""Grid-space kernels of the spectral dynamical core (``spectral_dynamics.F90``).

This module accretes the grid-space pieces of Isca's per-step
``spectral_dynamics`` tendency assembly. First up is :func:`four_in_one`, the
central "four in one" kernel (F90 ``subroutine four_in_one``, L1038-1112) that,
in a single vertical pass, forms:

1. the pressure-gradient force contributions to ``dt_ug`` / ``dt_vg``
   (``-R T grad(ln p)``),
2. the adiabatic (omega-alpha) heating contribution to ``dt_tg``,
3. the hybrid-coordinate vertical mass flux ``wg`` (interfaces) and the full-level
   vertical velocity ``wg_full`` (omega), and
4. the surface-pressure tendency accumulation into ``dt_psg``.

It depends only on ``rdgas`` / ``cp_air`` and the vertical-coordinate tables
``dpk = diff(pk)``, ``dbk = diff(bk)``, ``bk`` (from :mod:`jsca.dycore.vert_coordinate`)
— no spherical transforms — so it is a pure grid-space function.

Layout: level axis **last**. The 3-D grid fields ``divg`` / ``u_grid`` /
``v_grid`` / ``t_grid`` / ``ln_p_full`` / ``p_full`` are ``(..., K)``;
``ln_p_half`` is ``(..., K+1)``; ``p_surf`` / ``dx_psg`` / ``dy_psg`` / ``dt_psg``
are ``(...)``. ``dpk`` / ``dbk`` are ``(K,)`` and ``bk`` is ``(K+1,)``. Leading
axes are batched; jit/vmap/scan-safe.

Both Fortran vertical-difference options are ported: ``'simmons_and_burridge'``
(the default, F90 L1064) and ``'mcm'`` (F90 L1084). ``vert_difference_option`` is
a static argument.
"""

from __future__ import annotations

import jax.numpy as jnp

from jsca import constants

Array = jnp.ndarray


def _cumsum_exclusive(x: Array) -> Array:
    """``out[..., k] = sum_{j<k} x[..., j]`` (the running total *before* level k)."""
    incl = jnp.cumsum(x, axis=-1)
    return jnp.concatenate([jnp.zeros_like(x[..., :1]), incl[..., :-1]], axis=-1)


def four_in_one(
    divg: Array,
    u_grid: Array,
    v_grid: Array,
    t_grid: Array,
    p_surf: Array,
    ln_p_half: Array,
    ln_p_full: Array,
    p_full: Array,
    dx_psg: Array,
    dy_psg: Array,
    dt_psg: Array,
    dt_tg: Array,
    dt_ug: Array,
    dt_vg: Array,
    dpk: Array,
    dbk: Array,
    bk: Array,
    vert_difference_option: str = "simmons_and_burridge",
    rdgas: float = constants.RDGAS,
    cp_air: float = constants.CP_AIR,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Port of ``four_in_one`` (F90 L1038-1112).

    Returns ``(dt_psg, dt_tg, dt_ug, dt_vg, wg, wg_full)`` — the first four are the
    passed-in tendencies with this kernel's contributions accumulated;
    ``wg`` is ``(..., K+1)`` (interface mass flux, zero at top and surface) and
    ``wg_full`` is ``(..., K)``.
    """
    kappa = rdgas / cp_air  # F90 L1060
    k = t_grid.shape[-1]
    ps = p_surf[..., None]
    dxp = dx_psg[..., None]
    dyp = dy_psg[..., None]

    dp = dpk + dbk * ps  # layer thickness (F90 L1066/L1087)
    dmean = divg * dp + dbk * (u_grid * dxp + v_grid * dyp)  # F90 L1076/L1092
    dmean_excl = _cumsum_exclusive(dmean)  # running total before level k (Fortran dmean_tot)

    if vert_difference_option == "simmons_and_burridge":
        dp_inv = 1.0 / dp
        dlog_1 = ln_p_half[..., 1:] - ln_p_full  # F90 L1068
        dlog_2 = ln_p_full - ln_p_half[..., :-1]  # F90 L1069
        dlog_3 = ln_p_half[..., 1:] - ln_p_half[..., :-1]  # F90 L1070
        x1 = (bk[1:] * dlog_1 + bk[:-1] * dlog_2) * dp_inv  # F90 L1071
        x2 = x1 * dxp
        x3 = x1 * dyp
        x4 = (dmean_excl * dlog_3 + dmean * dlog_1) * dp_inv  # F90 L1077
    elif vert_difference_option == "mcm":
        ps_inv = 1.0 / ps
        x2 = dxp * ps_inv  # F90 L1088
        x3 = dyp * ps_inv
        x4 = (dmean_excl + 0.5 * dmean) / p_full  # F90 L1093
    else:
        raise ValueError(f"invalid vert_difference_option {vert_difference_option!r}")

    dt_ug = dt_ug - rdgas * t_grid * x2  # F90 L1074/L1090
    dt_vg = dt_vg - rdgas * t_grid * x3  # F90 L1075/L1091
    x5 = x4 - u_grid * x2 - v_grid * x3  # F90 L1078/L1094
    dt_tg = dt_tg - kappa * t_grid * x5  # F90 L1079/L1095
    wg_full = -x5 * p_full  # F90 L1080/L1096

    dmean_incl = jnp.cumsum(dmean, axis=-1)  # running total including level k
    dmean_total = dmean_incl[..., -1]  # grand total (Fortran dmean_tot after the loop)
    dt_psg = dt_psg - dmean_total  # F90 L1102

    # interface mass flux wg (F90 L1082/L1105-1109): wg[i] = -sum_{j<i} dmean + dmean_total*bk[i]
    # for interfaces i = 1..K-1; wg is zero at the top (0) and surface (K).
    zero = jnp.zeros_like(dmean[..., :1])
    inner = -dmean_incl[..., : k - 1] + dmean_total[..., None] * bk[1:k]
    wg = jnp.concatenate([zero, inner, zero], axis=-1)

    return dt_psg, dt_tg, dt_ug, dt_vg, wg, wg_full
