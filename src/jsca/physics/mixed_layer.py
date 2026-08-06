"""Slab-ocean surface energy balance — Isca's ``mixed_layer.F90``.

Faithful port of the Frierson slab-ocean step (``mixed_layer_nml: depth=2.5,
albedo_value=0.31, evaporation=.true.``; ``land_option='none'`` — pure ocean, no
q-flux, no prescribed SST). The mixed layer sits **between** the two vertical-
diffusion sweeps (:mod:`jsca.physics.vert_diff`): it closes the implicit surface
energy balance, updating the sea-surface temperature and the lowest-level T/q
increments the up sweep needs.

**The balance (F90 ``mixed_layer`` L637-746).** The surface flux of heat/moisture
depends implicitly on the (as-yet-unknown) lowest-level and surface temperatures,
so the update is done implicitly. With the vert_diff surface-coupling terms
(``dtmass``, ``dflux``, and the stored ``delta_t``/``delta_q``)::

    gamma_t = 1/(1 - dtmass·(dflux + dhdt_atm/cp))
    gamma_q = 1/(1 - dtmass·(dflux + dedq_atm))
    fn_t    = gamma_t·(delta_t + dtmass·flux_t/cp)   ;  en_t = gamma_t·dtmass·dhdt_surf/cp
    fn_q    = gamma_q·(delta_q + dtmass·flux_q)       ;  en_q = gamma_q·dtmass·dedt_surf

The net downward surface flux and its sensitivity to the surface temperature
(``alpha``/``beta`` terms) give an implicit heat-capacity update::

    corrected_flux    = -net_sw - lw_down + (flux_t + dhdt_atm·fn_t) + flux_r
                        + (flux_q + dedq_atm·fn_q)·HLV
    t_surf_dependence = (dhdt_surf + dhdt_atm·en_t) + drdt_surf
                        + (dedt_surf + dedq_atm·en_q)·HLV
    eff_heat_cap      = depth·RHO_CP + t_surf_dependence·dt
    delta_t_surf      = -corrected_flux·dt / eff_heat_cap ;  t_surf += delta_t_surf

Finally the lowest-level increments are corrected for the surface-temperature
change (F90 L745-746)::

    delta_t = fn_t + en_t·delta_t_surf ;  delta_q = fn_q + en_q·delta_t_surf

**Scope:** the Frierson ocean path (``evaporation=.true.``, no q-flux, no
prescribed/read SST, no bucket, no ice albedo, ``land_option='none'``). Pure
arithmetic — no documented deviation.

Layout: horizontal fields ``(...)``; consumes and returns a
:class:`jsca.physics.vert_diff.TriSurf` (its ``dflux`` is the shared
``-nu_n·(1-e_n1)`` that ``vert_diff`` stores for both heat and moisture).
"""
from __future__ import annotations

from dataclasses import dataclass

from jsca import constants
from jsca.physics.vert_diff import TriSurf

Array = "jax.Array"


@dataclass(frozen=True)
class MixedLayerParams:
    """Static ``mixed_layer_nml`` configuration (hashable; static jit arg)."""

    depth: float = 2.5           # slab-ocean depth (m)
    albedo: float = 0.31
    evaporation: bool = True


def mixed_layer_step(params: MixedLayerParams, t_surf, flux_t, flux_q, flux_r,
                     net_surf_sw_down, surf_lw_down, dhdt_surf, dedt_surf,
                     drdt_surf, dhdt_atm, dedq_atm, tri: TriSurf, dt):
    """One slab-ocean surface step.

    Args (all ``(...)`` horizontal, per :mod:`jsca.physics.surface_flux` /
    ``two_stream_gray_rad``): the sea-surface temperature, the surface fluxes
    (``flux_t`` sensible, ``flux_q`` evaporation, ``flux_r`` upward LW) and their
    derivatives, the downward SW/LW at the surface, and the vert_diff coupling in
    ``tri`` (``dtmass``, ``dflux``, ``delta_t``, ``delta_q``).

    Returns ``(t_surf_new, delta_t_surf, tri_out)`` — the updated SST, its
    increment, and ``tri`` with ``delta_t``/``delta_q`` updated for the up sweep.
    """
    inv_cp = 1.0 / constants.CP_AIR
    dtmass = tri.mu_delt_n   # Isca's Tri_surf%dtmass (surface layer mass * delt)
    dflux = tri.dflux

    gamma_t = 1.0 / (1.0 - dtmass * (dflux + dhdt_atm * inv_cp))
    gamma_q = 1.0 / (1.0 - dtmass * (dflux + dedq_atm))

    fn_t = gamma_t * (tri.delta_t + dtmass * flux_t * inv_cp)
    fn_q = gamma_q * (tri.delta_q + dtmass * flux_q)
    en_t = gamma_t * dtmass * dhdt_surf * inv_cp
    en_q = gamma_q * dtmass * dedt_surf

    alpha_t = flux_t * inv_cp + dhdt_atm * inv_cp * fn_t
    alpha_q = flux_q + dedq_atm * fn_q
    beta_t = dhdt_surf * inv_cp + dhdt_atm * inv_cp * en_t
    beta_q = dedt_surf + dedq_atm * en_q

    corrected_flux = -net_surf_sw_down - surf_lw_down + alpha_t * constants.CP_AIR + flux_r
    t_surf_dependence = beta_t * constants.CP_AIR + drdt_surf
    if params.evaporation:
        corrected_flux = corrected_flux + alpha_q * constants.HLV
        t_surf_dependence = t_surf_dependence + beta_q * constants.HLV

    heat_capacity = params.depth * constants.RHO_CP
    eff_heat_capacity = heat_capacity + t_surf_dependence * dt
    delta_t_surf = -corrected_flux * dt / eff_heat_capacity
    t_surf_new = t_surf + delta_t_surf

    delta_t = fn_t + en_t * delta_t_surf
    delta_q = fn_q + en_q * delta_t_surf if params.evaporation else tri.delta_q
    tri_out = tri._replace(delta_t=delta_t, delta_q=delta_q)
    return t_surf_new, delta_t_surf, tri_out
