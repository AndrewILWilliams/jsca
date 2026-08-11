"""Bulk surface fluxes — Isca's ``surface_flux.F90``, Frierson ocean path.

Faithful port of ``surface_flux_1d`` for the Frierson aquaplanet configuration
(``surface_flux_nml: do_simple=.true., use_virtual_temp=.false.,
old_dtaudv=.true.``) over ocean (``land=.false., seawater=.true.``, no bucket).
Given the lowest-model-level state and the sea-surface temperature, it returns
the bulk aerodynamic sensible-heat, evaporation and momentum-stress fluxes, plus
the implicit derivatives the surface energy balance needs.

The exchange coefficients come from Monin-Obukhov similarity
(:mod:`jsca.physics.monin_obukhov`); the saturation surface humidity from the
``do_simple`` Clausius-Clapeyron ``es`` (:mod:`jsca.physics.sat_vapor_pres`).

**Algorithm (F90 surface_flux_1d, ocean/do_simple/no-virtual-temp path):**

    q_sat  = eps*es(t_surf)/p_surf ;  q_surf0 = q_sat        (F90 L416-437)
    p_ratio = (p_surf/p_atm)**kappa                          (F90 L459)
    th_atm  = t_atm*p_ratio                                  (potential T)
    w_atm   = sqrt(u_dif^2 + v_dif^2 + gust^2)               (F90 L496)
    cd_m,cd_t,cd_q,u_star,b_star = mo_drag(t_atm*p_ratio, t_surf, ...)  (d608=0)
    rho     = p_atm/(rdgas*t_atm)
    flux_t  = cp_air*(cd_t*w_atm)*rho*(t_surf - th_atm)      (sensible heat)
    flux_q  = (cd_q*w_atm)*rho*(q_surf0 - q_atm)             (evaporation)
    flux_r  = sigma*t_surf^4                                 (upward LW)
    flux_u  = (cd_m*w_atm)*rho*u_dif ;  flux_v = ...*v_dif   (stress)

with the implicit derivatives ``dhdt_surf``/``dhdt_atm``/``dedt_surf``/
``dedq_atm``/``drdt_surf`` and (``old_dtaudv``) ``dtaudu_atm = dtaudv_atm =
-cd_m*w_atm*rho``. The 2 m / 10 m diagnostics (``temp_2m``, ``u_10m``, ``q_2m``,
``rh_2m``) use the Monin-Obukhov reference-height ratios (``mo_profile``).

Both ``use_virtual_temp`` settings are ported (Frierson uses False, ``column_test``
True) — see the ``surface_flux`` argument. **Scope otherwise:** the aquaplanet ocean
path only. Land/bucket evaporation, the NCAR ocean-flux override, and the
mixing-ratio / Raoult / alt-gustiness options are not ported (they are flagged
here). Deviation: only the documented ``sat_vapor_pres`` table-vs-closed-form
``es`` difference (~2e-7) enters, through ``q_sat`` and the 2 m diagnostics.

Layout: 1-D over horizontal points ``(N,)`` (fully vectorized).
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from jsca import constants
from jsca.physics.monin_obukhov import MOParams, mo_drag, mo_profile
from jsca.physics.sat_vapor_pres import saturation_vapor_pressure

Array = jnp.ndarray

_D622 = constants.RDGAS / constants.RVGAS   # eps
_DEL_TEMP = 0.1                              # F90 del_temp (finite-diff for dedt)
_ZREF_M = 10.0                              # 10 m winds
_ZREF_T = 2.0                               # 2 m temperature/humidity


class SurfaceFluxResult(NamedTuple):
    """Outputs of :func:`surface_flux` (all ``(N,)``); a JAX pytree."""

    flux_t: Array        # sensible heat flux (W/m^2)
    flux_q: Array        # evaporative water flux (kg/m^2/s)
    flux_r: Array        # upward longwave (W/m^2)
    flux_u: Array        # zonal momentum stress (Pa)
    flux_v: Array        # meridional momentum stress (Pa)
    cd_m: Array          # momentum exchange coefficient
    cd_t: Array          # heat exchange coefficient
    cd_q: Array          # moisture exchange coefficient
    w_atm: Array         # effective wind speed (incl. gust) (m/s)
    u_star: Array        # friction velocity (m/s)
    b_star: Array        # buoyancy scale
    q_star: Array        # moisture scale
    dhdt_surf: Array     # d(flux_t)/d(t_surf)
    dhdt_atm: Array      # d(flux_t)/d(t_atm)
    dedt_surf: Array     # d(flux_q)/d(t_surf)
    dedq_atm: Array      # d(flux_q)/d(q_atm)
    drdt_surf: Array     # d(flux_r)/d(t_surf)
    dtaudu_atm: Array    # d(stress)/d(u_atm)  (old_dtaudv)
    dtaudv_atm: Array    # d(stress)/d(v_atm)  (old_dtaudv)
    temp_2m: Array       # 2 m temperature (K)
    u_10m: Array         # 10 m zonal wind (m/s)
    q_2m: Array          # 2 m specific humidity (kg/kg)
    rh_2m: Array         # 2 m relative humidity
    depth_change_lh: Array  # bucket water removed by evaporation this step (m); 0 without bucket


def surface_flux(
    t_atm: Array, q_atm: Array, u_atm: Array, v_atm: Array,
    p_atm: Array, z_atm: Array, p_surf: Array, t_surf: Array,
    u_surf: Array, v_surf: Array,
    rough_mom: Array, rough_heat: Array, rough_moist: Array, gust: Array,
    q_surf_in: Array,
    mo_params: MOParams = MOParams(),
    use_virtual_temp: bool = False,
    bucket: bool = False,
    bucket_depth: Array | None = None,
    max_bucket_depth_land: float = 2.0,
    land: Array | None = None,
    dt: float = 0.0,
) -> SurfaceFluxResult:
    """Bulk ocean surface fluxes (Frierson do_simple path).

    Args are 1-D ``(N,)``: lowest-level air state (``t_atm``, ``q_atm``,
    ``u_atm``, ``v_atm``, ``p_atm``, ``z_atm``), surface pressure/temperature,
    surface winds (0 on an aquaplanet), roughness lengths, gustiness, and the
    incoming ``q_surf`` (used only by the 2 m diagnostic). ``rough_scale`` is
    taken equal to ``rough_mom`` (the driver's setting), so the orographic drag
    rescaling is the identity.

    ``use_virtual_temp`` (``surface_flux_nml``; Frierson False, column_test True)
    selects the virtual-temperature correction to the surface-layer stability
    (F90 L461-464, L573): the Monin-Obukhov drag then sees the *virtual* potential
    temperatures ``thv = th*(1 + d608*q)`` and the density uses the virtual
    temperature, so buoyancy accounts for moisture. With it False, ``d608 = 0`` and
    everything reduces to the actual/potential-temperature path.
    """
    kappa = constants.KAPPA          # = rdgas/cp_air
    # d608 = rvgas/rdgas - 1 (= d378/d622); zeroed when use_virtual_temp is off
    # (F90 surface_flux_init L928-937).
    d608 = (constants.RVGAS / constants.RDGAS - 1.0) if use_virtual_temp else 0.0
    # --- surface saturation humidity (do_simple) and its temperature derivative ---
    e_sat = saturation_vapor_pressure(t_surf)
    e_sat1 = saturation_vapor_pressure(t_surf + _DEL_TEMP)
    q_sat = _D622 * e_sat / p_surf
    q_sat1 = _D622 * e_sat1 / p_surf
    # ocean: saturated surface. Bucket model (F90 surface_flux L448-455): a dry
    # surface (empty bucket) has q_surf0 = q_atm, killing the humidity gradient (so
    # no evaporation); a wet surface stays saturated. q_surf0 feeds the M-O
    # virtual-temperature stability below, so the switch is applied here.
    if bucket:
        q_surf0 = jnp.where(bucket_depth <= 0.0, q_atm, q_sat)
    else:
        q_surf0 = q_sat

    # --- Monin-Obukhov drag: virtual potential temperatures (F90 L461-464) ---
    p_ratio = (p_surf / p_atm) ** kappa
    th_atm = t_atm * p_ratio                       # potential T (used by the fluxes)
    tv_atm = t_atm * (1.0 + d608 * q_atm)          # virtual T (used by rho)
    thv_atm = th_atm * (1.0 + d608 * q_atm)        # virtual potential T (= tv_atm*p_ratio)
    thv_surf = t_surf * (1.0 + d608 * q_surf0)     # surface virtual (potential) T

    u_dif = u_surf - u_atm
    v_dif = v_surf - v_atm
    w_atm = jnp.sqrt(u_dif * u_dif + v_dif * v_dif + gust * gust)

    cd_m, cd_t, cd_q, u_star, b_star = mo_drag(
        mo_params, thv_atm, thv_surf, z_atm, rough_mom, rough_heat, rough_moist, w_atm)

    ex_del_m, ex_del_h, ex_del_q = mo_profile(
        mo_params, _ZREF_M, _ZREF_T, z_atm, rough_mom, rough_heat, rough_moist,
        u_star, b_star)

    # 2 m / 10 m diagnostics (F90 L521-555)
    temp_2m = t_surf + (t_atm - t_surf) * ex_del_h
    u_10m = u_atm * ex_del_m
    q_2m = q_surf_in + (q_atm - q_surf_in) * ex_del_q
    q_sat_2m = _D622 * saturation_vapor_pressure(temp_2m) / p_surf
    rh_2m = q_2m / q_sat_2m

    # orographic drag rescaling (identity here: rough_scale = rough_mom, F90 L566)
    # cd_m *= (log(z/rough_mom+1)/log(z/rough_mom+1))**2 == 1
    drag_t = cd_t * w_atm
    drag_q = cd_q * w_atm
    drag_m = cd_m * w_atm
    rho = p_atm / (constants.RDGAS * tv_atm)    # virtual T (= t_atm when d608=0)

    # sensible heat (F90 L575-579)
    rho_drag_t = constants.CP_AIR * drag_t * rho
    flux_t = rho_drag_t * (t_surf - th_atm)
    dhdt_surf = rho_drag_t
    dhdt_atm = -rho_drag_t * p_ratio

    # evaporation. Ocean/no-bucket path is F90 L636-641; the bucket path
    # (F90 surface_flux L587-643) adds a beta ramp over land and a hard cap.
    rho_drag_q = drag_q * rho
    if bucket:
        thresh = max_bucket_depth_land * 0.75            # 0.75*max: ramp threshold
        # beta ramp: full evaporation over ocean or a wet-enough land bucket;
        # linearly reduced over land once the bucket drops below 0.75*max.
        below = land & (bucket_depth < thresh)
        beta = jnp.where(below, bucket_depth / thresh, 1.0)
        flux_q = beta * rho_drag_q * (q_surf0 - q_atm)
        # cap: evaporation cannot remove more water than the bucket holds this step
        depth_change_lh = flux_q * dt / constants.DENS_H2O
        over = (flux_q > 0.0) & (bucket_depth < depth_change_lh)
        flux_q = jnp.where(over, bucket_depth * constants.DENS_H2O / dt, flux_q)
        depth_change_lh = flux_q * dt / constants.DENS_H2O
        # implicit derivatives: zero when the bucket is empty, else beta-ramped
        empty = bucket_depth <= 0.0
        dedt_full = rho_drag_q * (q_sat1 - q_sat) / _DEL_TEMP
        dedt_surf = jnp.where(empty, 0.0, beta * dedt_full)
        dedq_atm = jnp.where(empty, 0.0, -rho_drag_q)
    else:
        flux_q = rho_drag_q * (q_surf0 - q_atm)
        dedt_surf = rho_drag_q * (q_sat1 - q_sat) / _DEL_TEMP
        dedq_atm = -rho_drag_q
        depth_change_lh = jnp.zeros_like(flux_q)

    q_star = flux_q / (u_star * rho)

    # upward longwave (F90 L654-655)
    flux_r = constants.STEFAN * t_surf ** 4
    drdt_surf = 4.0 * constants.STEFAN * t_surf ** 3

    # momentum stress (F90 L658-660, L685-689 old_dtaudv)
    rho_drag_m = drag_m * rho
    flux_u = rho_drag_m * u_dif
    flux_v = rho_drag_m * v_dif
    dtaudu_atm = -rho_drag_m
    dtaudv_atm = -rho_drag_m

    return SurfaceFluxResult(
        flux_t=flux_t, flux_q=flux_q, flux_r=flux_r, flux_u=flux_u, flux_v=flux_v,
        cd_m=cd_m, cd_t=cd_t, cd_q=cd_q, w_atm=w_atm,
        u_star=u_star, b_star=b_star, q_star=q_star,
        dhdt_surf=dhdt_surf, dhdt_atm=dhdt_atm, dedt_surf=dedt_surf,
        dedq_atm=dedq_atm, drdt_surf=drdt_surf,
        dtaudu_atm=dtaudu_atm, dtaudv_atm=dtaudv_atm,
        temp_2m=temp_2m, u_10m=u_10m, q_2m=q_2m, rh_2m=rh_2m,
        depth_change_lh=depth_change_lh,
    )
