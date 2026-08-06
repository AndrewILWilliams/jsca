"""Frierson grey (two-stream) radiation — Isca's ``rad_scheme = 'frierson'``.

Faithful port of the Frierson branches of
``src/atmos_param/two_stream_gray_rad/two_stream_gray_rad.F90`` — the grey
radiation scheme the Frierson (2006) moist aquaplanet uses
(``two_stream_gray_rad_nml: rad_scheme='frierson', do_seasonal=.false.,
atm_abs=0.2``). The atmosphere is a grey (wavelength-independent) absorber with a
prescribed optical-depth profile; there is no wavelength dependence and no cloud.

**Shortwave (``two_stream_gray_rad_down``, F90 L452-492):** a fixed
top-of-atmosphere insolation with the Frierson p2 profile (no diurnal/seasonal
cycle), attenuated downward through an optical depth that grows as a power of
pressure::

    p2         = (1 - 3 sin^2 lat) / 4
    insol      = 0.25 * S0 * (1 + del_sol*p2 + del_sw*sin lat)          (F90 L454-455)
    sw_tau_0   = (1 - sw_diff*sin^2 lat) * atm_abs                       (F90 L482)
    sw_tau(k)  = sw_tau_0 * (p_half(k)/p_std)**solar_exponent            (F90 L486)
    sw_down(k) = insol * exp(-sw_tau(k))                                 (F90 L491)

**Longwave (down: F90 L574-594; up: F90 L690-695):** a two-stream grey
integration with a source ``b = sigma T^4``. The optical depth mixes a linear
(well-mixed-gas) and a power-law (water-vapour-like) term::

    lw_tau_0   = (ir_tau_eq + (ir_tau_pole - ir_tau_eq) sin^2 lat) * odp
    lw_tau(k)  = lw_tau_0 * (linear_tau*p_half(k)/p_std
                            + (1-linear_tau)*(p_half(k)/p_std)**wv_exponent)
    dtrans(k)  = exp(-(lw_tau(k+1) - lw_tau(k)))                (layer transmissivity)
    lw_down(k+1) = lw_down(k)*dtrans(k) + b(k)*(1 - dtrans(k)),  lw_down(0) = 0
    lw_up(k)     = lw_up(k+1)*dtrans(k) + b(k)*(1 - dtrans(k)),  lw_up(K) = sigma T_surf^4

**Heating (F90 L710-728):** upward SW is a constant column reflection
``albedo * sw_down(surface)``; the net (positive-up) flux divergence gives the
radiative heating rate::

    net_surf_sw_down = sw_down(K) * (1 - albedo)
    surf_lw_down     = lw_down(K)
    rad_flux         = (lw_up - lw_down) + (sw_up - sw_down)
    tdt_rad(k)       = diabatic_acce * (rad_flux(k+1) - rad_flux(k))
                       * grav / (cp_air * (p_half(k+1) - p_half(k)))

The scheme is split into a **down** pass (SW + LW-down → surface fluxes, before
the surface temperature is known) and an **up** pass (LW-up + heating, using the
updated ``t_surf``), matching Isca so the surface (mixed-layer) update can slot
between them. No documented deviations: pure arithmetic (``exp``/power laws), so
fixtures hit the tight tolerance.

Layout: column physics, **level axis last**; ``t`` is ``(..., K)``, ``p_half``
is ``(..., K+1)``, ``lat`` is ``(...)``. Leading axes are batched.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jsca import constants

Array = jnp.ndarray

#: Standard reference pressure ``pstd_mks`` (Pa) — Isca ``constants.F90`` L263.
PSTD_MKS = 101325.0


@dataclass(frozen=True)
class GrayRadParams:
    """Static Frierson grey-radiation configuration (hashable; static jit arg).

    Defaults are the Isca ``two_stream_gray_rad_nml`` defaults, with the Frierson
    aquaplanet override ``atm_abs = 0.2`` (the namelist default is 0.0).
    """

    solar_constant: float = 1360.0
    del_sol: float = 1.4
    del_sw: float = 0.0
    ir_tau_eq: float = 6.0
    ir_tau_pole: float = 1.5
    atm_abs: float = 0.2       # Frierson override (namelist default 0.0)
    odp: float = 1.0
    sw_diff: float = 0.0
    linear_tau: float = 0.1
    wv_exponent: float = 4.0
    solar_exponent: float = 4.0
    diabatic_acce: float = 1.0


@dataclass(frozen=True)
class RadDownState:
    """Intermediate state carried from the down pass to the up pass."""

    lw_dtrans: Array   # (..., K) layer LW transmissivities
    b: Array           # (..., K) LW source sigma*T^4 at full levels
    sw_down: Array     # (..., K+1) downward SW flux at half levels
    lw_down: Array     # (..., K+1) downward LW flux at half levels


def _lw_down_integral(dtrans: Array, b: Array) -> Array:
    """Downward LW: ``lw_down(k+1) = lw_down(k)*dtrans(k) + b(k)*(1-dtrans(k))``.

    ``lw_down(0) = 0``. Returns ``(..., K+1)``. Level axis last; the recurrence
    runs top→surface as a ``lax.scan``.
    """
    dt_f = jnp.moveaxis(dtrans, -1, 0)
    b_f = jnp.moveaxis(b, -1, 0)

    def step(carry, x):
        dtk, bk = x
        new = carry * dtk + bk * (1.0 - dtk)
        return new, new

    init = jnp.zeros(dtrans.shape[:-1])
    _, outs = jax.lax.scan(step, init, (dt_f, b_f))          # outs[k] = lw_down(k+1)
    full = jnp.concatenate([init[None], outs], axis=0)       # (K+1, ...)
    return jnp.moveaxis(full, 0, -1)


def _lw_up_integral(dtrans: Array, b: Array, b_surf: Array) -> Array:
    """Upward LW: ``lw_up(k) = lw_up(k+1)*dtrans(k) + b(k)*(1-dtrans(k))``.

    ``lw_up(K) = b_surf``. Returns ``(..., K+1)``. A reverse ``lax.scan``
    (surface→top); outputs stack back in ascending level order.
    """
    dt_f = jnp.moveaxis(dtrans, -1, 0)
    b_f = jnp.moveaxis(b, -1, 0)

    def step(carry, x):
        dtk, bk = x
        new = carry * dtk + bk * (1.0 - dtk)
        return new, new

    _, outs = jax.lax.scan(step, b_surf, (dt_f, b_f), reverse=True)  # outs[k] = lw_up(k)
    full = jnp.concatenate([outs, b_surf[None]], axis=0)             # (K+1, ...)
    return jnp.moveaxis(full, 0, -1)


def gray_rad_down(params: GrayRadParams, lat: Array, p_half: Array, t: Array,
                  albedo: Array):
    """Down pass: SW + downward LW → surface fluxes (F90 ``..._down``).

    Returns ``(net_surf_sw_down, surf_lw_down, state)`` — the surface downward
    SW absorbed ``(1-albedo)*sw_down(sfc)`` and downward LW ``(...)`` [W/m^2],
    and a :class:`RadDownState` for the up pass.
    """
    ph = p_half / PSTD_MKS                                 # normalized pressure (..., K+1)

    # --- shortwave ---
    p2 = (1.0 - 3.0 * jnp.sin(lat) ** 2) / 4.0
    insol = 0.25 * params.solar_constant * (
        1.0 + params.del_sol * p2 + params.del_sw * jnp.sin(lat))
    sw_tau_0 = (1.0 - params.sw_diff * jnp.sin(lat) ** 2) * params.atm_abs
    sw_tau = sw_tau_0[..., None] * ph ** params.solar_exponent
    sw_down = insol[..., None] * jnp.exp(-sw_tau)          # (..., K+1)

    # --- longwave source + optical depth ---
    b = constants.STEFAN * t ** 4                          # (..., K)
    lw_tau_0 = (params.ir_tau_eq
                + (params.ir_tau_pole - params.ir_tau_eq) * jnp.sin(lat) ** 2) * params.odp
    lw_tau = lw_tau_0[..., None] * (
        params.linear_tau * ph + (1.0 - params.linear_tau) * ph ** params.wv_exponent)
    lw_dtrans = jnp.exp(-(lw_tau[..., 1:] - lw_tau[..., :-1]))   # (..., K)

    lw_down = _lw_down_integral(lw_dtrans, b)
    surf_lw_down = lw_down[..., -1]
    net_surf_sw_down = sw_down[..., -1] * (1.0 - albedo)

    return net_surf_sw_down, surf_lw_down, RadDownState(lw_dtrans, b, sw_down, lw_down)


def gray_rad_up(params: GrayRadParams, p_half: Array, t_surf: Array, albedo: Array,
                state: RadDownState):
    """Up pass: upward LW + net-flux heating (F90 ``..._up``).

    Returns ``(tdt_rad, olr, net_lw_surf)`` — the radiative heating rate
    ``(..., K)`` [K/s], the outgoing longwave at TOA and the net upward LW at the
    surface ``(...)`` [W/m^2].
    """
    b_surf = constants.STEFAN * t_surf ** 4
    lw_up = _lw_up_integral(state.lw_dtrans, state.b, b_surf)

    sw_down_sfc = state.sw_down[..., -1]                   # SW reflected is column-constant
    sw_up = albedo[..., None] * sw_down_sfc[..., None] * jnp.ones_like(state.sw_down)

    lw_flux = lw_up - state.lw_down                        # (..., K+1)
    sw_flux = sw_up - state.sw_down
    rad_flux = lw_flux + sw_flux

    dflux = rad_flux[..., 1:] - rad_flux[..., :-1]
    dp = p_half[..., 1:] - p_half[..., :-1]
    tdt_rad = params.diabatic_acce * dflux * constants.GRAV / (constants.CP_AIR * dp)

    olr = lw_up[..., 0]
    net_lw_surf = lw_flux[..., -1]
    return tdt_rad, olr, net_lw_surf


def two_stream_gray_rad(params: GrayRadParams, lat: Array, p_half: Array,
                        t: Array, t_surf: Array, albedo: Array):
    """Full grey-radiation step (down then up), convenience wrapper.

    Returns ``(tdt_rad, net_surf_sw_down, surf_lw_down, olr, net_lw_surf)``.
    """
    net_sw, lw_dn, state = gray_rad_down(params, lat, p_half, t, albedo)
    tdt_rad, olr, net_lw_surf = gray_rad_up(params, p_half, t_surf, albedo, state)
    return tdt_rad, net_sw, lw_dn, olr, net_lw_surf
