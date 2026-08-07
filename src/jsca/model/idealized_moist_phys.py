"""Frierson ``idealized_moist_phys`` column-physics driver — the per-step stack.

Assembles the individually fixture-validated Frierson column-physics modules into
one pure tendency function, in Isca's exact call order (``idealized_moist_phys.F90``
``subroutine idealized_moist_phys``, L819-1337). This is the **assembly** step of
the moist port (roadmap item 11): each component below is already golden-validated
against the real Fortran to machine precision; this module wires them together.

Call order and time levels (F90 line refs), Frierson options
(``convection_scheme='SIMPLE_BETTS_MILLER'``, ``two_stream_gray``, ``turb``,
``mixed_layer_bc``, ``do_simple``, ``land_option='none'``):

1. **Convection** ``qe_moist_convection`` (F90 L862) — on the *previous* level;
   returns temperature/humidity **increments** ``dT``/``dq`` (not rates) and rain.
   ``dt_tg += dT/delta_t``; ``dt_qg += dq/delta_t`` (F90 L877/971-972). The
   post-convection profile ``tg_tmp = t_prev + dT`` feeds condensation (L873-874).
2. **Large-scale condensation** ``lscale_cond`` (F90 L981) on ``tg_tmp/qg_tmp``;
   increments added the same way (F90 L987/994-995).
3. **Grey radiation, down** ``gray_rad_down`` (F90 L1054) — surface SW/LW down,
   independent of the surface temperature; uses ``t_prev``, current ``p_half``.
4. **Surface flux** ``surface_flux`` (F90 L1077) — sensible/latent/momentum
   fluxes and their implicit derivatives from the *previous* lowest-level state
   and ``t_surf``.
5. **Grey radiation, up** ``gray_rad_up`` (F90 L1156) — radiative heating into
   ``dt_tg`` (uses ``t_surf`` and the down-pass state).
6. **Rayleigh sponge** ``rayleigh_sponge`` (F90 L1228) — top-of-model drag on the
   *previous* winds into ``dt_ug``/``dt_vg`` (+ conserving heating into ``dt_tg``).
7. **Boundary-layer diffusivity** ``diffusivity`` (F90 L1242, via
   ``vert_turb_driver``) — ``K_m``/``K_t`` from the surface-layer ``u*``/``b*``.
8. **Implicit vertical diffusion, down** ``vert_diff_down`` (F90 L1292) — the
   momentum solve + the ``TriSurf`` surface coupling for T/q.
9. **Slab ocean** ``mixed_layer_step`` (F90 L1309) — the surface energy balance,
   updating ``t_surf`` (uses ``dt_real``, **not** ``delta_t`` — F90 L1317).
10. **Implicit vertical diffusion, up** ``vert_diff_up`` (F90 L1330) — finishes
    the T/q diffusion using the updated surface.

Layout: grid columns level-last ``(..., nlat, nlon, K)``; half levels ``K+1``;
horizontal surface fields ``(..., nlat, nlon)``. Aquaplanet: ``land=.false.``,
``z_surf=0``, ocean at rest (``u_surf=v_surf=0``), a saturated ocean surface.

**Validation status.** Every module here is golden-fixture-validated on its own.
The *assembly* (call order + tendency bookkeeping) is checked by the stability /
conservation smoke test in ``tests/test_idealized_moist_phys.py``; an end-to-end
golden step fixture from an instrumented Isca Frierson run (like the dry-core
``spectral_dynamics_step`` fixture) is the remaining validation and needs a full
Isca build — see ``docs/frierson_roadmap.md`` item 11.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax.numpy as jnp

from jsca.physics.damping_driver import DampingDriverParams, rayleigh_sponge
from jsca.physics.diffusivity import DiffusivityParams, diffusivity
from jsca.physics.lscale_cond import lscale_cond
from jsca.physics.mixed_layer import MixedLayerParams, mixed_layer_step
from jsca.physics.monin_obukhov import MOParams
from jsca.physics.qe_moist_convection import qe_moist_convection
from jsca.physics.surface_flux import surface_flux
from jsca.physics.two_stream_gray_rad import GrayRadParams, gray_rad_down, gray_rad_up
from jsca.physics.vert_diff import vert_diff_down, vert_diff_up

Array = jnp.ndarray


@dataclass(frozen=True)
class FriersonPhysicsParams:
    """Static configuration bundle for :func:`idealized_moist_phys` (Frierson).

    Groups the per-module parameter objects plus the surface roughness / gustiness
    / albedo scalars from the Frierson namelists (``frierson_test_case.py``).
    """

    gray_rad: GrayRadParams = field(default_factory=GrayRadParams)
    mo: MOParams = field(default_factory=MOParams)
    diff: DiffusivityParams = field(default_factory=DiffusivityParams)
    mixed_layer: MixedLayerParams = field(default_factory=MixedLayerParams)
    damping: DampingDriverParams | None = None  # from damping_driver_init(pref)
    roughness_mom: float = 3.21e-5
    roughness_heat: float = 3.21e-5
    roughness_moist: float = 3.21e-5
    gust_const: float = 0.0        # constant_gust (Frierson: 0.0)
    albedo: float = 0.31


class MoistPhysicsOutput(NamedTuple):
    """Result of one column-physics step: the accumulated tendencies, the updated
    slab-ocean temperature, precipitation (kg/m^2/s), and boundary-layer height.
    A ``NamedTuple`` so it is a valid JAX pytree (jit/scan-returnable)."""

    dt_ug: Array
    dt_vg: Array
    dt_tg: Array
    dt_qg: Array
    t_surf: Array
    precip: Array
    pbl_height: Array


def idealized_moist_phys(
    params: FriersonPhysicsParams,
    lat2d: Array,
    lon2d: Array,
    # previous-level grid state (level-last)
    u_prev: Array,
    v_prev: Array,
    t_prev: Array,
    q_prev: Array,
    p_half_prev: Array,
    p_full_prev: Array,
    # current-level pressures / geometry
    p_half_cur: Array,
    p_full_cur: Array,
    z_full_cur: Array,
    t_surf: Array,
    delta_t: float,
    dt_real: float,
) -> MoistPhysicsOutput:
    """One Frierson column-physics step (F90 ``idealized_moist_phys`` L819-1337).

    Returns a :class:`MoistPhysicsOutput`. Tendencies accumulate from zero here;
    the caller adds them to the dynamical-core tendencies before the leapfrog.
    ``z_full_cur`` is the geopotential height above the surface (``z_surf = 0`` for
    the aquaplanet). Winds/temperature/humidity are the *previous* leapfrog level
    (the driver's ``ug/vg/tg/grid_tracers(...,previous)``); pressures split into
    previous (convection/condensation) and current (radiation/surface/diffusion).
    """
    if params.damping is None:
        raise ValueError("params.damping must be set via damping_driver_init(pref)")

    shape = t_prev.shape
    dt_ug = jnp.zeros(shape)
    dt_vg = jnp.zeros(shape)
    dt_tg = jnp.zeros(shape)
    dt_qg = jnp.zeros(shape)

    # --- 1. convection (previous level; returns increments) --- F90 L862-877
    rain_c, dtemp_c, dq_c, _cflag = qe_moist_convection(
        t_prev, q_prev, p_full_prev, p_half_prev, delta_t)
    tg_tmp = t_prev + dtemp_c
    qg_tmp = q_prev + dq_c
    dt_tg = dt_tg + dtemp_c / delta_t
    dt_qg = dt_qg + dq_c / delta_t
    precip = rain_c / delta_t

    # --- 2. large-scale condensation on the post-convection profile --- F90 L981
    rain_l, dtemp_l, dq_l = lscale_cond(tg_tmp, qg_tmp, p_full_prev, p_half_prev)
    dt_tg = dt_tg + dtemp_l / delta_t
    dt_qg = dt_qg + dq_l / delta_t
    precip = precip + rain_l / delta_t

    # --- 3. grey radiation, down (surface SW/LW down; needs t_prev) --- F90 L1054
    albedo = jnp.broadcast_to(jnp.asarray(params.albedo), lat2d.shape)
    net_surf_sw_down, surf_lw_down, rad_state = gray_rad_down(
        params.gray_rad, lat2d, p_half_cur, t_prev, albedo)

    # --- 4. surface fluxes (previous lowest level + t_surf) --- F90 L1077
    z_atm = z_full_cur[..., -1]              # height of the lowest level (z_surf = 0)
    p_atm = p_full_cur[..., -1]
    p_surf = p_half_cur[..., -1]             # surface pressure (bottom half level)
    zero_s = jnp.zeros(lat2d.shape)
    rough_mom = jnp.broadcast_to(jnp.asarray(params.roughness_mom), lat2d.shape)
    rough_heat = jnp.broadcast_to(jnp.asarray(params.roughness_heat), lat2d.shape)
    rough_moist = jnp.broadcast_to(jnp.asarray(params.roughness_moist), lat2d.shape)
    gust = jnp.broadcast_to(jnp.asarray(params.gust_const), lat2d.shape)
    sf = surface_flux(
        t_prev[..., -1], q_prev[..., -1], u_prev[..., -1], v_prev[..., -1],
        p_atm, z_atm, p_surf, t_surf, zero_s, zero_s,
        rough_mom, rough_heat, rough_moist, gust, zero_s, params.mo)

    # --- 5. grey radiation, up (radiative heating into dt_tg) --- F90 L1156
    tdt_rad, _olr, _net_lw = gray_rad_up(
        params.gray_rad, p_half_cur, t_surf, albedo, rad_state)
    dt_tg = dt_tg + tdt_rad

    # --- 6. Rayleigh sponge on the previous winds --- F90 L1228
    udt_s, vdt_s, tdt_s = rayleigh_sponge(params.damping, delta_t, p_full_cur, u_prev, v_prev)
    dt_ug = dt_ug + udt_s
    dt_vg = dt_vg + vdt_s
    dt_tg = dt_tg + tdt_s

    # --- 7. boundary-layer diffusivity (K profiles) --- F90 L1242
    z_half_cur = _half_from_full_heights(z_full_cur)
    diff_m, diff_t, pbl_h = diffusivity(
        params.diff, t_prev, q_prev, u_prev, v_prev, z_full_cur, z_half_cur,
        sf.u_star, sf.b_star, params.mo)

    # --- 8. implicit vertical diffusion, down (momentum + TriSurf) --- F90 L1292
    dt_ug, dt_vg, _diss, tri = vert_diff_down(
        delta_t, u_prev, v_prev, t_prev, q_prev, diff_m, diff_t,
        p_half_cur, p_full_cur, z_full_cur, sf.flux_u, sf.flux_v,
        sf.dtaudu_atm, sf.dtaudv_atm, dt_ug, dt_vg, dt_tg, dt_qg)

    # --- 9. slab-ocean surface energy balance (uses dt_real, not delta_t) --- F90 L1309
    t_surf_new, _dts, tri = mixed_layer_step(
        params.mixed_layer, t_surf, sf.flux_t, sf.flux_q, sf.flux_r,
        net_surf_sw_down, surf_lw_down, sf.dhdt_surf, sf.dedt_surf, sf.drdt_surf,
        sf.dhdt_atm, sf.dedq_atm, tri, dt_real)

    # --- 10. implicit vertical diffusion, up (finish T/q) --- F90 L1330
    dt_tg, dt_qg = vert_diff_up(delta_t, tri)

    return MoistPhysicsOutput(
        dt_ug=dt_ug, dt_vg=dt_vg, dt_tg=dt_tg, dt_qg=dt_qg,
        t_surf=t_surf_new, precip=precip, pbl_height=pbl_h)


def _half_from_full_heights(z_full: Array) -> Array:
    """Half-level heights ``(..., K+1)`` from full-level heights by simple midpoint
    interpolation, with the surface at 0 and the top extrapolated. Used only by the
    boundary-layer ``diffusivity`` Richardson-number profile, which references
    ``z_half`` differences; the aquaplanet surface is at ``z = 0``."""
    interior = 0.5 * (z_full[..., :-1] + z_full[..., 1:])
    top = z_full[..., :1] + (z_full[..., :1] - interior[..., :1])
    surf = jnp.zeros_like(z_full[..., :1])
    return jnp.concatenate([top, interior, surf], axis=-1)
