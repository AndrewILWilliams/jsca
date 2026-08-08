"""The Held & Suarez (1994) benchmark: dynamical core + HS forcing, time-stepped.

Assembles the fixture-validated dynamical-core tendencies
(:func:`jsca.dycore.compute_tendencies`) with the Held-Suarez physics forcing
(:func:`jsca.physics.hs_forcing`) into a leapfrog/RAW-filtered time step, applies
the mass + energy conservation corrections, and integrates from rest with
``lax.scan``. This is the end-to-end dry benchmark used to validate the whole
port against the canonical HS94 climatology (eddy-driven midlatitude jets, the
tropopause temperature structure).

Time levels: each prognostic spectral field carries a length-2 axis, slot 0 =
previous, slot 1 = current. Each step forms the ``future`` via the leapfrog
(overwriting slot 0), then rolls the slots so previous/current are 0/1 again.

Layout follows :mod:`jsca.dycore.dynamics`: spectral ``(m, n, K, 2)`` /
``(m, n, 2)``; grid diagnostics ``(nlat, nlon, K)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jsca import constants
from jsca.dycore.dynamics import DynamicsParams, _to_last, build_dynamics_params, compute_tendencies
from jsca.dycore.global_integral import mass_weighted_global_integral
from jsca.dycore.implicit import build_wave_matrices
from jsca.dycore.leapfrog import leapfrog
from jsca.dycore.press_and_geopot import pressure_variables
from jsca.dycore.spectral_dynamics import energy_correction, mass_correction
from jsca.grid.transforms import (
    area_weighted_global_mean,
    grid_to_spectral,
    spectral_to_grid,
    uv_grid_from_vor_div,
)
from jsca.physics.hs_forcing import HsForcingParams, hs_forcing, hs_forcing_init

Array = jnp.ndarray


@dataclass(frozen=True)
class HeldSuarezModel:
    dyn: DynamicsParams
    hs: HsForcingParams
    wave_matrix: Array  # for delta_t = 2 dt (the leapfrog interval)
    wave_matrix_cold: Array  # for delta_t = dt (the cold-start forward step)
    lat2d: Array  # (nlat, nlon)
    dt: float  # physical timestep
    delta_t: float  # leapfrog interval (2 dt)
    robert_coeff: float
    raw_filter_coeff: float
    nlat: int
    nlon: int


def build_held_suarez(
    num_fourier: int = 21,
    nlat: int | None = None,
    nlon: int | None = None,
    num_levels: int = 20,
    dt: float = 600.0,  # advective CFL at T21 with ~30 m/s jets needs dt <~ 900 s
    robert_coeff: float = 0.04,
    raw_filter_coeff: float = 1.0,  # Isca default (RAW off; 0.53 "appears unstable" per Isca)
    forcing: HsForcingParams | None = None,
    **dyn_kwargs,
) -> HeldSuarezModel:
    """Build a Held-Suarez model. ``dt`` is the physical timestep; the leapfrog
    interval is ``2 dt``. ``forcing`` supplies the Newtonian-relaxation /
    Rayleigh-friction configuration (the hook the object API in :mod:`jsca.api`
    uses to thread notebook-set knobs through); ``None`` uses the HS94 defaults."""
    nlat = nlat or (2 * num_fourier + 2)
    nlon = nlon or (4 * num_fourier + 4)
    dyn = build_dynamics_params(num_fourier, nlat, nlon, num_levels, **dyn_kwargs)
    hs = forcing if forcing is not None else hs_forcing_init()
    delta_t = 2.0 * dt
    wave_matrix = build_wave_matrices(dyn.implicit, delta_t)
    wave_matrix_cold = build_wave_matrices(dyn.implicit, dt)
    lat = np.arcsin(np.asarray(dyn.transforms.sin_lat))[:, None] * np.ones((1, nlon))
    return HeldSuarezModel(
        dyn=dyn, hs=hs, wave_matrix=wave_matrix, wave_matrix_cold=wave_matrix_cold,
        lat2d=jnp.asarray(lat), dt=dt, delta_t=delta_t, robert_coeff=robert_coeff,
        raw_filter_coeff=raw_filter_coeff, nlat=nlat, nlon=nlon,
    )


def initial_state(m: HeldSuarezModel, temperature: float = 264.0, surface_press: float = 1.0e5,
                  seed: int = 0, perturb: float = 1.0e-4):
    """Resting isothermal state with a tiny random temperature perturbation to
    break symmetry. Returns ``(vors, divs, ts, ln_ps)`` with a length-2 time axis
    (previous == current == the initial condition)."""
    tf = m.dyn.transforms
    k = m.dyn.num_levels
    tg = np.full((k, m.nlat, m.nlon), temperature)
    rng = np.random.default_rng(seed)
    tg = tg + perturb * rng.standard_normal(tg.shape)
    # triangular-truncate the initial fields to l <= M (Isca's get_initial_fields
    # produces triangular fields; grid_to_spectral keeps the l=M+1 storage row)
    mp = tf.mask_prognostic
    ts = _to_last(grid_to_spectral(tf, jnp.asarray(tg)))  # (m, n, K)
    ts = jnp.where(mp[..., None], ts, 0.0)
    ln_ps = grid_to_spectral(tf, jnp.full((m.nlat, m.nlon), float(np.log(surface_press))))
    ln_ps = jnp.where(mp, ln_ps, 0.0)
    zero = jnp.zeros_like(ts)
    stack = lambda x: jnp.stack([x, x], axis=-1)  # noqa: E731
    return stack(zero), stack(zero), stack(ts), stack(ln_ps)


def _grid_from_spectral(m: HeldSuarezModel, vors, divs, ts, ln_ps, slot):
    """Grid ``(u, v, T, ps)`` (level-last ``(nlat, nlon, K)``) at a time slot."""
    tf = m.dyn.transforms
    vc = jnp.moveaxis(vors[..., slot], -1, 0)  # (K, m, n)
    dc = jnp.moveaxis(divs[..., slot], -1, 0)
    tc = jnp.moveaxis(ts[..., slot], -1, 0)
    u_l, v_l = uv_grid_from_vor_div(tf, vc, dc)
    u, v = _to_last(u_l), _to_last(v_l)
    t = _to_last(spectral_to_grid(tf, tc))
    ps = jnp.exp(spectral_to_grid(tf, ln_ps[..., slot]))  # (nlat, nlon)
    return u, v, t, ps


def step(m: HeldSuarezModel, state, delta_t: float | None = None, wave_matrix=None):
    """One leapfrog step: HS forcing -> dynamics tendencies -> leapfrog -> mass +
    energy corrections -> roll time slots. ``state = (vors, divs, ts, ln_ps)``.

    ``delta_t``/``wave_matrix`` default to the leapfrog interval ``2 dt``; the
    cold-start step passes ``dt`` (see :func:`integrate`), matching Isca's
    ``spectral_dynamics`` (``delta_t = dt_real`` when ``previous == current``,
    ``2 dt_real`` after).
    """
    vors, divs, ts, ln_ps = state
    prev, cur, fut = 0, 1, 0  # future overwrites the previous slot
    dyn, tf = m.dyn, m.dyn.transforms
    dtl = m.delta_t if delta_t is None else delta_t
    wm = m.wave_matrix if wave_matrix is None else wave_matrix

    # grid state at current (pressure) and previous (winds/temperature for the
    # forcing, and the reference means for the corrections)
    _, _, _, ps_c = _grid_from_spectral(m, vors, divs, ts, ln_ps, cur)
    u_p, v_p, t_p, ps_p = _grid_from_spectral(m, vors, divs, ts, ln_ps, prev)
    p_half, _, p_full, _ = pressure_variables(dyn.pk, dyn.bk, ps_c, dyn.vert_difference_option)

    # Held-Suarez physics tendencies. Pressure is at the CURRENT level but the
    # winds/temperature are at the PREVIOUS level -- Isca's driver passes
    # ug/vg/tg(previous) (atmosphere.F90 L304-311). Applying Rayleigh friction to
    # the lagged level is essential for leapfrog stability; using the current
    # level feeds the computational mode (verified: current-level friction is a
    # ~8% error against the Fortran and destabilises the high-resolution run).
    udt, vdt, tdt, _ = hs_forcing(m.hs, m.lat2d, p_half, p_full, u_p, v_p, t_p, u_p, v_p, dtl)

    # reference global means from the previous level advanced by the physics
    # forcing over delta_t (initialize_corrections, spectral_dynamics.F90 L1373-1379)
    mean_sp_prev = area_weighted_global_mean(tf, ps_p)
    energy_p = (0.5 * ((u_p + udt * dtl) ** 2 + (v_p + vdt * dtl) ** 2)
                + constants.CP_AIR * (t_p + tdt * dtl))
    mean_en_prev = mass_weighted_global_integral(tf, dyn.pk, dyn.bk, energy_p, ps_p)

    # dynamical tendencies (with the physics forcing folded in)
    dvor, ddiv, dts, dlnps = compute_tendencies(
        dyn, vors, divs, ts, ln_ps, dtl, wm, prev, cur, udt, vdt, tdt,
    )

    # leapfrog + RAW filter (future overwrites slot 0)
    rc, raw = m.robert_coeff, m.raw_filter_coeff
    vors = leapfrog(vors, dvor, prev, cur, fut, dtl, rc, raw)
    divs = leapfrog(divs, ddiv, prev, cur, fut, dtl, rc, raw)
    ts = leapfrog(ts, dts, prev, cur, fut, dtl, rc, raw)
    ln_ps = leapfrog(ln_ps, dlnps, prev, cur, fut, dtl, rc, raw)

    # mass + energy conservation corrections on the future ((0,0) coefficients)
    u_f, v_f, t_f, ps_f = _grid_from_spectral(m, vors, divs, ts, ln_ps, fut)
    lnps00 = jnp.real(ln_ps[0, 0, fut])
    ts00 = jnp.real(ts[0, 0, :, fut])
    ps_c2, lnps00_new, _ = mass_correction(tf, ps_f, lnps00, mean_sp_prev)
    _, ts00_new, _ = energy_correction(
        tf, dyn.pk, dyn.bk, t_f, ts00, u_f, v_f, ps_c2, mean_en_prev, mean_sp_prev,
    )
    ln_ps = ln_ps.at[0, 0, fut].set(lnps00_new + 0.0j)
    ts = ts.at[0, 0, :, fut].set(ts00_new + 0.0j)

    # roll slots: new previous = old current (slot 1), new current = future (slot 0)
    roll = lambda a: jnp.stack([a[..., cur], a[..., fut]], axis=-1)  # noqa: E731
    return (roll(vors), roll(divs), roll(ts), roll(ln_ps))


def integrate(m: HeldSuarezModel, state, n_steps: int, sample_every: int = 0,
              cold_start: bool = False):
    """Integrate ``n_steps`` with ``lax.scan``. If ``sample_every > 0``, also
    returns the grid ``(u, T, ps)`` at the current level every ``sample_every``
    steps (for climatology accumulation).

    ``cold_start=True`` runs the first step as Isca's start-up forward step
    (``delta_t = dt`` with the ``dt`` wave matrix) before the leapfrog scan; pass
    it only on the first integrate from the resting :func:`initial_state`.
    """
    jstep = jax.jit(lambda s: step(m, s))

    if cold_start and n_steps > 0:
        state = jax.jit(lambda s: step(m, s, m.dt, m.wave_matrix_cold))(state)
        n_steps -= 1

    if sample_every <= 0:
        def body(s, _):
            return jstep(s), None
        state, _ = jax.lax.scan(body, state, None, length=n_steps)
        return state

    n_out = n_steps // sample_every

    def outer(carry, _):
        s = carry
        for _ in range(sample_every):
            s = jstep(s)
        vors, divs, ts, ln_ps = s
        u, _, t, ps = _grid_from_spectral(m, vors, divs, ts, ln_ps, 1)
        return s, (u, t, ps)

    state, samples = jax.lax.scan(outer, state, None, length=n_out)
    return state, samples
