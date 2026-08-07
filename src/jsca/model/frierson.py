"""The Frierson (2006) moist aquaplanet: dynamical core + moist physics, stepped.

Extends the dry Held-Suarez stepping (:mod:`jsca.model.held_suarez`) with the two
things the moist model adds: the **``sphum`` grid tracer** (advected by
:func:`jsca.dycore.update_grid_tracer` with the PPM vertical scheme, conserved by
:func:`jsca.dycore.water_correction`) and the **slab-ocean ``t_surf``** (updated by
the mixed layer inside the column physics). The per-step column physics is the
assembled :func:`jsca.model.idealized_moist_phys.idealized_moist_phys` driver.

Per-step flow (Isca ``atmosphere``/``spectral_dynamics`` + ``idealized_moist_phys``):

1. grid state (u, v, T, q, ps) at previous & current from the spectral / grid
   prognostics;
2. **column physics** ``idealized_moist_phys`` on the previous level → grid
   tendencies ``dt_ug/dt_vg/dt_tg`` (folded into the dynamics) and ``dt_qg`` (the
   humidity physics tendency for the tracer), plus the updated ``t_surf`` and
   precipitation;
3. **spectral dynamics** ``compute_tendencies`` with the physics forcing → the
   damped, semi-implicit spectral tendencies **and** the current-level vertical
   mass flux ``wg`` / winds the tracer advection needs;
4. **leapfrog + RAW** the spectral prognostics; mass + energy corrections;
5. **grid tracer step** ``update_grid_tracer`` (physics tendency + horizontal +
   PPM vertical advection + Robert/RAW filter), then the global
   ``water_correction`` on the future humidity;
6. roll the time slots; carry the new ``t_surf``.

Time levels: spectral fields and the grid humidity carry a length-2 axis (slot 0
previous, slot 1 current); the future overwrites slot 0 then the slots roll.

**Validation status.** Every kernel here is golden-fixture-validated against Isca;
this end-to-end *stepping* is gated by the stability/conservation smoke test in
``tests/test_frierson.py``. The machine-precision step-fixture and the
climatology-vs-Isca comparison (roadmap item 11c) need a pinned Isca Frierson
reference (a full Isca build) — see ``docs/frierson_roadmap.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jsca import constants
from jsca.dycore.dynamics import DynamicsParams, _to_last, build_dynamics_params, compute_tendencies
from jsca.dycore.fv_advection import FvAdvectionParams, fv_advection_init
from jsca.dycore.global_integral import mass_weighted_global_integral
from jsca.dycore.implicit import build_wave_matrices
from jsca.dycore.leapfrog import leapfrog
from jsca.dycore.press_and_geopot import (
    compute_geopotential,
    pressure_variables,
)
from jsca.dycore.spectral_dynamics import (
    energy_correction,
    mass_correction,
    update_grid_tracer,
    water_correction,
)
from jsca.dycore.vert_advection import FINITE_VOLUME_PARABOLIC
from jsca.grid.transforms import (
    area_weighted_global_mean,
    grid_to_spectral,
    spectral_to_grid,
    uv_grid_from_vor_div,
)
from jsca.model.idealized_moist_phys import (
    FriersonPhysicsParams,
    idealized_moist_phys,
)
from jsca.physics.damping_driver import damping_driver_init

Array = jnp.ndarray

# Frierson pure-sigma 25-level coordinate (frierson_test_case.py: pk = 0, bk given)
FRIERSON_BK = np.array([
    0.000000, 0.0117665, 0.0196679, 0.0315244, 0.0485411, 0.0719344, 0.1027829,
    0.1418581, 0.1894648, 0.2453219, 0.3085103, 0.3775033, 0.4502789, 0.5244989,
    0.5977253, 0.6676441, 0.7322627, 0.7900587, 0.8400683, 0.8819111, 0.9157609,
    0.9422770, 0.9625127, 0.9778177, 0.9897489, 1.0000000])
FRIERSON_PK = np.zeros(26)


@dataclass(frozen=True)
class FriersonModel:
    dyn: DynamicsParams
    phys: FriersonPhysicsParams
    fv: FvAdvectionParams          # A-grid metrics for tracer horizontal advection
    wave_matrix: Array             # delta_t = 2 dt
    wave_matrix_cold: Array        # delta_t = dt (cold-start forward step)
    lat2d: Array
    lon2d: Array
    dt: float
    delta_t: float
    robert_coeff: float
    raw_filter_coeff: float
    nlat: int
    nlon: int


def build_frierson(
    num_fourier: int = 42,
    nlat: int | None = None,
    nlon: int | None = None,
    dt: float = 720.0,             # Frierson dt_atmos
    robert_coeff: float = 0.03,    # Frierson robert_coeff
    raw_filter_coeff: float = 1.0,
    mixed_layer_depth: float = 2.5,
    albedo: float = 0.31,
    **dyn_kwargs,
) -> FriersonModel:
    """Build a Frierson moist model at T``num_fourier`` with the 25-level pure-sigma
    coordinate. ``dt`` is the physical timestep; the leapfrog interval is ``2 dt``."""
    num_levels = len(FRIERSON_BK) - 1
    nlat = nlat or (2 * num_fourier + 2)
    nlon = nlon or (4 * num_fourier + 4)
    dyn = build_dynamics_params(
        num_fourier, nlat, nlon, num_levels,
        pk=FRIERSON_PK, bk=FRIERSON_BK, **dyn_kwargs)

    # reference full-level pressures for the sponge depth (pressure_variables at PSTD)
    p_half_1d, _, p_full_1d, _ = pressure_variables(
        FRIERSON_PK, FRIERSON_BK, jnp.asarray(constants.PSTD_MKS), "simmons_and_burridge")
    from jsca.physics.mixed_layer import MixedLayerParams
    phys = FriersonPhysicsParams(
        mixed_layer=MixedLayerParams(depth=mixed_layer_depth, albedo=albedo),
        damping=damping_driver_init(np.asarray(p_full_1d)),
        albedo=albedo,
    )

    # A-grid latitude cell edges: midpoints of sin(lat) with poles at +/-1
    # (aquaplanet convention; the exact Gaussian-boundary form is pinned once the
    # golden Frierson fixture exists -- roadmap item 11c).
    sin_lat = np.asarray(dyn.transforms.sin_lat)
    sin_edges = np.concatenate([[-1.0], 0.5 * (sin_lat[1:] + sin_lat[:-1]), [1.0]])
    lat_edges = np.arcsin(np.clip(sin_edges, -1.0, 1.0))
    fv = fv_advection_init(nlon, lat_edges, degrees_lon=360.0)

    delta_t = 2.0 * dt
    lat = np.arcsin(sin_lat)[:, None] * np.ones((1, nlon))
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False)[None, :] * np.ones((nlat, 1))
    return FriersonModel(
        dyn=dyn, phys=phys, fv=fv,
        wave_matrix=build_wave_matrices(dyn.implicit, delta_t),
        wave_matrix_cold=build_wave_matrices(dyn.implicit, dt),
        lat2d=jnp.asarray(lat), lon2d=jnp.asarray(lon), dt=dt, delta_t=delta_t,
        robert_coeff=robert_coeff, raw_filter_coeff=raw_filter_coeff, nlat=nlat, nlon=nlon,
    )


def initial_state(m: FriersonModel, temperature: float = 280.0, surface_press: float = 1.0e5,
                  t_surf: float = 285.0, humidity: float = 1.0e-6, seed: int = 0,
                  perturb: float = 1.0e-4):
    """Resting isothermal, near-dry state with a small temperature perturbation and
    a warm uniform SST. Returns ``(vors, divs, ts, ln_ps, qg, t_surf)``; the
    spectral fields and ``qg`` carry a length-2 time axis (previous == current)."""
    tf = m.dyn.transforms
    k = m.dyn.num_levels
    rng = np.random.default_rng(seed)
    tg = np.full((k, m.nlat, m.nlon), temperature)
    tg = tg + perturb * rng.standard_normal((k, m.nlat, m.nlon))
    mp = tf.mask_prognostic
    ts = _to_last(grid_to_spectral(tf, jnp.asarray(tg)))
    ts = jnp.where(mp[..., None], ts, 0.0)
    lnps_grid = jnp.full((m.nlat, m.nlon), float(np.log(surface_press)))
    ln_ps = jnp.where(mp, grid_to_spectral(tf, lnps_grid), 0.0)
    zero = jnp.zeros_like(ts)
    qg = jnp.full((m.nlat, m.nlon, k), humidity)
    stack = lambda x: jnp.stack([x, x], axis=-1)  # noqa: E731
    tsurf = jnp.full((m.nlat, m.nlon), t_surf)
    return stack(zero), stack(zero), stack(ts), stack(ln_ps), stack(qg), tsurf


def _grid_from_spectral(m: FriersonModel, vors, divs, ts, ln_ps, slot):
    """Grid ``(u, v, T, ps)`` (level-last) at a time slot."""
    tf = m.dyn.transforms
    u_l, v_l = uv_grid_from_vor_div(tf, jnp.moveaxis(vors[..., slot], -1, 0),
                                    jnp.moveaxis(divs[..., slot], -1, 0))
    t = _to_last(spectral_to_grid(tf, jnp.moveaxis(ts[..., slot], -1, 0)))
    ps = jnp.exp(spectral_to_grid(tf, ln_ps[..., slot]))
    return _to_last(u_l), _to_last(v_l), t, ps


def step(m: FriersonModel, state, delta_t: float | None = None, wave_matrix=None):
    """One moist leapfrog step. ``state = (vors, divs, ts, ln_ps, qg, t_surf)``."""
    vors, divs, ts, ln_ps, qg, t_surf = state
    prev, cur, fut = 0, 1, 0
    dyn, tf = m.dyn, m.dyn.transforms
    dtl = m.delta_t if delta_t is None else delta_t
    wm = m.wave_matrix if wave_matrix is None else wave_matrix

    u_p, v_p, t_p, ps_p = _grid_from_spectral(m, vors, divs, ts, ln_ps, prev)
    _, _, _, ps_c = _grid_from_spectral(m, vors, divs, ts, ln_ps, cur)
    q_p, q_c = qg[..., prev], qg[..., cur]

    ph_p, lph_p, pf_p, lpf_p = pressure_variables(dyn.pk, dyn.bk, ps_p, dyn.vert_difference_option)
    ph_c, lph_c, pf_c, lpf_c = pressure_variables(dyn.pk, dyn.bk, ps_c, dyn.vert_difference_option)
    # geopotential heights above the surface (aquaplanet: surf geopotential = 0).
    # The full AND half level heights are needed: the boundary-layer diffusivity's
    # Richardson profile references z_half, so a midpoint approximation is not
    # faithful (verified against Isca: it corrupts the diffusion tendency).
    phi_full, phi_half = compute_geopotential(dyn.pk, t_p, lph_c, lpf_c, dyn.surf_geopotential)
    z_full_c = phi_full / constants.GRAV
    z_half_c = phi_half / constants.GRAV

    # --- column physics on the previous level ---
    # gust is Isca's stateful vert_turb gustiness; Frierson sets constant_gust=0, so
    # the steady-state value the diffusivity path uses is 0 (the cold-start step-1
    # transient value of 1.0 m/s is not reproduced -- a documented start-up
    # difference, negligible for the climatology).
    gust = m.phys.gust_const * jnp.ones(m.lat2d.shape)
    phys = idealized_moist_phys(
        m.phys, m.lat2d, m.lon2d, u_p, v_p, t_p, q_p, ph_p, pf_p, ph_c, pf_c, z_full_c,
        z_half_c, t_surf, gust, dtl, m.dt)

    # reference means for the mass/energy corrections (previous advanced by physics)
    mean_sp_prev = area_weighted_global_mean(tf, ps_p)
    energy_p = (0.5 * ((u_p + phys.dt_ug * dtl) ** 2 + (v_p + phys.dt_vg * dtl) ** 2)
                + constants.CP_AIR * (t_p + phys.dt_tg * dtl))
    mean_en_prev = mass_weighted_global_integral(tf, dyn.pk, dyn.bk, energy_p, ps_p)
    mean_water_prev = mass_weighted_global_integral(tf, dyn.pk, dyn.bk, q_p, ps_p)

    # --- spectral dynamics (physics forcing folded in) + tracer diagnostics ---
    dvor, ddiv, dts, dlnps, (wg, ug_c, vg_c, ph_c2) = compute_tendencies(
        dyn, vors, divs, ts, ln_ps, dtl, wm, prev, cur,
        phys.dt_ug, phys.dt_vg, phys.dt_tg, return_diagnostics=True)

    rc, raw = m.robert_coeff, m.raw_filter_coeff
    vors = leapfrog(vors, dvor, prev, cur, fut, dtl, rc, raw)
    divs = leapfrog(divs, ddiv, prev, cur, fut, dtl, rc, raw)
    ts = leapfrog(ts, dts, prev, cur, fut, dtl, rc, raw)
    ln_ps = leapfrog(ln_ps, dlnps, prev, cur, fut, dtl, rc, raw)

    # mass + energy conservation corrections (future (0,0) coefficients)
    u_f, v_f, t_f, ps_f = _grid_from_spectral(m, vors, divs, ts, ln_ps, fut)
    ps_f2, lnps00_new, _ = mass_correction(tf, ps_f, jnp.real(ln_ps[0, 0, fut]), mean_sp_prev)
    _, ts00_new, _ = energy_correction(
        tf, dyn.pk, dyn.bk, t_f, jnp.real(ts[0, 0, :, fut]), u_f, v_f, ps_f2,
        mean_en_prev, mean_sp_prev)
    ln_ps = ln_ps.at[0, 0, fut].set(lnps00_new + 0.0j)
    ts = ts.at[0, 0, :, fut].set(ts00_new + 0.0j)

    # --- grid tracer (sphum): advection + RAW filter, then water conservation ---
    q_cur_new, q_fut, _pf = update_grid_tracer(
        q_p, q_c, phys.dt_qg, ug_c, vg_c, wg, ph_c2, dtl, rc, raw, m.fv,
        scheme=FINITE_VOLUME_PARABOLIC)
    # global water-conservation correction on the future humidity (uses future ps)
    q_fut, _ = water_correction(
        tf, dyn.pk, dyn.bk, q_fut, ps_f2, pf_c, mean_water_prev,
        water_correction_limit=200.0e2)  # Frierson: 200 hPa

    # roll slots (new previous = old current; new current = future)
    roll = lambda a: jnp.stack([a[..., cur], a[..., fut]], axis=-1)  # noqa: E731
    qg = jnp.stack([q_cur_new, q_fut], axis=-1)
    return (roll(vors), roll(divs), roll(ts), roll(ln_ps), qg, phys.t_surf)


def integrate(m: FriersonModel, state, n_steps: int, cold_start: bool = False):
    """Integrate ``n_steps`` with ``lax.scan``. ``cold_start`` runs the first step
    as Isca's forward start-up step (``delta_t = dt``)."""
    jstep = jax.jit(lambda s: step(m, s))
    if cold_start and n_steps > 0:
        state = jax.jit(lambda s: step(m, s, m.dt, m.wave_matrix_cold))(state)
        n_steps -= 1
    state, _ = jax.lax.scan(lambda s, _: (jstep(s), None), state, None, length=n_steps)
    return state
