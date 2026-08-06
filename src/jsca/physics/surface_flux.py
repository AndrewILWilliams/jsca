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

**Scope:** the aquaplanet ocean path only. Land/bucket evaporation, the NCAR
ocean-flux override, the mixing-ratio / Raoult / alt-gustiness options, and
``use_virtual_temp=.true.`` (``d608 != 0``) are not ported (they are flagged
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


def surface_flux(
    t_atm: Array, q_atm: Array, u_atm: Array, v_atm: Array,
    p_atm: Array, z_atm: Array, p_surf: Array, t_surf: Array,
    u_surf: Array, v_surf: Array,
    rough_mom: Array, rough_heat: Array, rough_moist: Array, gust: Array,
    q_surf_in: Array,
    mo_params: MOParams = MOParams(),
) -> SurfaceFluxResult:
    """Bulk ocean surface fluxes (Frierson do_simple path).

    Args are 1-D ``(N,)``: lowest-level air state (``t_atm``, ``q_atm``,
    ``u_atm``, ``v_atm``, ``p_atm``, ``z_atm``), surface pressure/temperature,
    surface winds (0 on an aquaplanet), roughness lengths, gustiness, and the
    incoming ``q_surf`` (used only by the 2 m diagnostic). ``rough_scale`` is
    taken equal to ``rough_mom`` (the driver's setting), so the orographic drag
    rescaling is the identity.
    """
    kappa = constants.KAPPA          # = rdgas/cp_air
    # --- surface saturation humidity (do_simple) and its temperature derivative ---
    e_sat = saturation_vapor_pressure(t_surf)
    e_sat1 = saturation_vapor_pressure(t_surf + _DEL_TEMP)
    q_sat = _D622 * e_sat / p_surf
    q_sat1 = _D622 * e_sat1 / p_surf
    q_surf0 = q_sat                  # ocean: saturated surface

    # --- Monin-Obukhov drag (d608 = 0, so thv = potential/actual T) ---
    p_ratio = (p_surf / p_atm) ** kappa
    th_atm = t_atm * p_ratio
    thv_atm = th_atm                 # use_virtual_temp = False
    thv_surf = t_surf

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
    rho = p_atm / (constants.RDGAS * t_atm)     # tv_atm = t_atm (d608=0)

    # sensible heat (F90 L575-579)
    rho_drag_t = constants.CP_AIR * drag_t * rho
    flux_t = rho_drag_t * (t_surf - th_atm)
    dhdt_surf = rho_drag_t
    dhdt_atm = -rho_drag_t * p_ratio

    # evaporation (F90 L636-641, ocean/no-bucket)
    rho_drag_q = drag_q * rho
    flux_q = rho_drag_q * (q_surf0 - q_atm)
    dedt_surf = rho_drag_q * (q_sat1 - q_sat) / _DEL_TEMP
    dedq_atm = -rho_drag_q

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
    )
