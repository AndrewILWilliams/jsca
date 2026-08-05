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
from jsca.dycore.global_integral import mass_weighted_global_integral
from jsca.grid.transforms import area_weighted_global_mean

Array = jnp.ndarray

_SQRT2 = jnp.sqrt(jnp.asarray(2.0))


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


# --------------------------------------------------------------------------- #
# conservation corrections (compute_corrections, F90 L1213-1302)
# --------------------------------------------------------------------------- #
#
# After the leapfrog produces the future state, tiny truncation errors let the
# global dry-air mass and total energy drift. compute_corrections restores them
# by rescaling the surface pressure so the global-mean surface pressure matches
# the reference (mass), and by adding a uniform temperature offset so the global
# total energy matches the reference (energy). Both are single global-mean
# adjustments; the matching change is also made to the (0, 0) spectral
# coefficient of ln(ps) / T so the grid and spectral states stay consistent.
#
# Layout: jsca grid convention, ``(..., nlat, nlon)`` for ``psg`` and
# ``(..., nlat, nlon, K)`` for ``ug``/``vg``/``tg`` (the global integrals weight
# by Gaussian latitude, so latitude must be axis -2 / -3). The Fortran
# ``(lon, lat, lev)`` fixtures transpose lon<->lat.
#
# The wet-model water correction (F90 L1247-1281, incl. the MiMA pressure-limit
# variant) is a follow-up: it needs the humidity tracer and its spectral
# representation. ``do_water_correction`` raises here.


def mass_correction(
    params, psg: Array, ln_ps_00: Array, mean_surf_press_previous: Array
) -> tuple[Array, Array, Array]:
    """Global dry-mass correction — port of F90 L1226-1234.

    Rescales ``psg`` so its area-weighted global mean equals
    ``mean_surf_press_previous``; returns ``(psg, ln_ps_00, factor)`` with the
    matching additive tweak ``+ sqrt(2) log(factor)`` applied to the ``(0,0)``
    spectral coefficient of ``ln(ps)``.
    """
    mean_tmp = area_weighted_global_mean(params, psg)
    factor = mean_surf_press_previous / mean_tmp
    return factor[..., None, None] * psg, ln_ps_00 + _SQRT2 * jnp.log(factor), factor


def energy_correction(
    params,
    pk: Array,
    bk: Array,
    tg: Array,
    ts_00: Array,
    ug: Array,
    vg: Array,
    psg: Array,
    mean_energy_previous: Array,
    mean_surf_press_previous: Array,
    grav: float = constants.GRAV,
    cp_air: float = constants.CP_AIR,
) -> tuple[Array, Array, Array]:
    """Global total-energy correction — port of F90 L1237-1243.

    Adds a uniform temperature offset so the mass-weighted global-mean total
    energy ``0.5(u^2+v^2) + cp T`` matches ``mean_energy_previous``. ``psg`` must
    already be mass-corrected (F90 applies mass first). Returns
    ``(tg, ts_00, temperature_correction)`` with ``+ sqrt(2) dT`` applied to the
    ``(0,0)`` spectral coefficient of ``T``.
    """
    energy = 0.5 * (ug**2 + vg**2) + cp_air * tg  # (..., nlat, nlon, K)
    mean_energy_tmp = mass_weighted_global_integral(params, pk, bk, energy, psg, grav)
    dtemp = grav * (mean_energy_previous - mean_energy_tmp) / (cp_air * mean_surf_press_previous)
    return tg + dtemp[..., None, None, None], ts_00 + _SQRT2 * dtemp[..., None], dtemp


def compute_corrections(
    params,
    pk: Array,
    bk: Array,
    psg: Array,
    ug: Array,
    vg: Array,
    tg: Array,
    ln_ps_00: Array,
    ts_00: Array,
    mean_surf_press_previous: Array,
    mean_energy_previous: Array,
    do_mass_correction: bool = True,
    do_energy_correction: bool = True,
    do_water_correction: bool = False,
    grav: float = constants.GRAV,
    cp_air: float = constants.CP_AIR,
) -> tuple[Array, Array, Array, Array]:
    """Mass + energy conservation corrections — port of ``compute_corrections``
    (F90 L1213-1302, dry path). Returns the corrected
    ``(psg, tg, ln_ps_00, ts_00)``.

    ``ln_ps_00`` is the ``(0,0)`` spectral coefficient of ``ln(ps)`` (a scalar per
    batch); ``ts_00`` is the ``(0,0)`` coefficient of ``T`` (``(..., K)``). These
    are real for a real field. ``do_water_correction`` (wet models) is not ported.
    """
    if do_water_correction:
        raise NotImplementedError("water correction (wet models) is a follow-up")
    if do_mass_correction:
        psg, ln_ps_00, _ = mass_correction(params, psg, ln_ps_00, mean_surf_press_previous)
    if do_energy_correction:
        tg, ts_00, _ = energy_correction(
            params, pk, bk, tg, ts_00, ug, vg, psg,
            mean_energy_previous, mean_surf_press_previous, grav, cp_air,
        )
    return psg, tg, ln_ps_00, ts_00
