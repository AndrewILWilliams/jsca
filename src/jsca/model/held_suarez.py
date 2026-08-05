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
    wave_matrix: Array  # for delta_t = 2 dt
    lat2d: Array  # (nlat, nlon)
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
    **dyn_kwargs,
) -> HeldSuarezModel:
    """Build a Held-Suarez model. ``dt`` is the physical timestep; the leapfrog
    interval is ``2 dt``."""
    nlat = nlat or (2 * num_fourier + 2)
    nlon = nlon or (4 * num_fourier + 4)
    dyn = build_dynamics_params(num_fourier, nlat, nlon, num_levels, **dyn_kwargs)
    hs = hs_forcing_init()
    delta_t = 2.0 * dt
    wave_matrix = build_wave_matrices(dyn.implicit, delta_t)
    lat = np.arcsin(np.asarray(dyn.transforms.sin_lat))[:, None] * np.ones((1, nlon))
    return HeldSuarezModel(
        dyn=dyn, hs=hs, wave_matrix=wave_matrix, lat2d=jnp.asarray(lat),
        delta_t=delta_t, robert_coeff=robert_coeff, raw_filter_coeff=raw_filter_coeff,
        nlat=nlat, nlon=nlon,
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
    ts = _to_last(grid_to_spectral(tf, jnp.asarray(tg)))  # (m, n, K)
    ln_ps = grid_to_spectral(tf, jnp.full((m.nlat, m.nlon), float(np.log(surface_press))))
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


def step(m: HeldSuarezModel, state):
    """One leapfrog step: HS forcing -> dynamics tendencies -> leapfrog -> mass +
    energy corrections -> roll time slots. ``state = (vors, divs, ts, ln_ps)``."""
    vors, divs, ts, ln_ps = state
    prev, cur, fut = 0, 1, 0  # future overwrites the previous slot
    dyn, tf = m.dyn, m.dyn.transforms

    # grid state at current (for HS forcing) and previous (winds for friction, and
    # the reference means for the corrections)
    u_c, v_c, t_c, ps_c = _grid_from_spectral(m, vors, divs, ts, ln_ps, cur)
    u_p, v_p, t_p, ps_p = _grid_from_spectral(m, vors, divs, ts, ln_ps, prev)
    p_half, _, p_full, _ = pressure_variables(dyn.pk, dyn.bk, ps_c, dyn.vert_difference_option)

    # Held-Suarez physics tendencies (grid)
    udt, vdt, tdt, _ = hs_forcing(m.hs, m.lat2d, p_half, p_full, u_c, v_c, t_c, u_p, v_p, m.delta_t)

    # reference global means from the previous level (initialize_corrections)
    mean_sp_prev = area_weighted_global_mean(tf, ps_p)
    from jsca.dycore.global_integral import mass_weighted_global_integral
    energy_p = 0.5 * (u_p**2 + v_p**2) + constants.CP_AIR * t_p
    mean_en_prev = mass_weighted_global_integral(tf, dyn.pk, dyn.bk, energy_p, ps_p)

    # dynamical tendencies (with the physics forcing folded in)
    dvor, ddiv, dts, dlnps = compute_tendencies(
        dyn, vors, divs, ts, ln_ps, m.delta_t, m.wave_matrix, prev, cur, udt, vdt, tdt,
    )

    # leapfrog + RAW filter (future overwrites slot 0)
    rc, raw, dtl = m.robert_coeff, m.raw_filter_coeff, m.delta_t
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


def integrate(m: HeldSuarezModel, state, n_steps: int, sample_every: int = 0):
    """Integrate ``n_steps`` with ``lax.scan``. If ``sample_every > 0``, also
    returns the grid ``(u, T, ps)`` at the current level every ``sample_every``
    steps (for climatology accumulation)."""
    jstep = jax.jit(lambda s: step(m, s))

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
