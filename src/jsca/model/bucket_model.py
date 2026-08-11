"""Land + bucket-hydrology moist model — the ``land_option='input'`` companion to
:mod:`jsca.model.frierson`.

This is the Frierson moist aquaplanet stepping (dynamical core + grey radiation +
the assembled column physics) with the three land pieces switched on:

* a static **land mask** (:func:`jsca.model.land.continents_land_mask`) that scales
  the mixed-layer heat capacity and albedo over land
  (:func:`jsca.physics.mixed_layer.land_heat_capacity` /
  :func:`jsca.physics.mixed_layer.land_albedo`);
* the **bucket evaporation switch** in the surface flux (dry land → no
  evaporation, β-ramped below capacity — :mod:`jsca.physics.surface_flux`);
* a prognostic **soil-moisture reservoir** ``bucket_depth`` stepped every timestep
  by :func:`jsca.physics.bucket.bucket_step` from the precipitation and evaporation
  the column physics returns.

It reproduces Isca's ``exp/test_cases/bucket_hydrology`` configuration **run with
grey radiation** (the shipped test case uses RRTM; grey radiation is the jsca
equivalent requested here) and a **realistic-continents** land mask rather than the
square block Isca ships. The dynamics, leapfrog, corrections and tracer step are
byte-for-byte the validated :mod:`jsca.model.frierson` machinery — only the extra
land/bucket wiring and the ``bucket_depth`` prognostic are new.

**State** ``(vors, divs, ts, ln_ps, qg, t_surf, bucket_depth)``: the six Frierson
prognostics plus ``bucket_depth`` ``(nlat, nlon, 2)`` (leapfrog time level last,
slot 0 previous, slot 1 current — same convention as ``qg``). The bucket is marched
on the *same* ``previous/current/future`` indices as the atmosphere, with its own
Robert/RAW coefficients (``robert_bucket=0.04``, ``raw_bucket=0.53``).

**Validation status.** Every kernel is golden-fixture-validated on its own (grid /
dynamics / physics / bucket stepping / land mask); this end-to-end land stepping is
gated here by a stability + bucket-water-budget smoke test
(``tests/test_bucket_model.py``). The climatology-vs-Isca comparison at T21 is the
next stage and needs a pinned Isca bucket run.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jsca import constants
from jsca.dycore.dynamics import (
    _to_last,
    build_dynamics_params,
    compute_tendencies,
)
from jsca.dycore.fv_advection import fv_advection_init
from jsca.dycore.global_integral import mass_weighted_global_integral
from jsca.dycore.implicit import build_wave_matrices
from jsca.dycore.leapfrog import leapfrog
from jsca.dycore.press_and_geopot import compute_geopotential, pressure_variables
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
)
from jsca.model.frierson import FriersonModel, _grid_from_spectral
from jsca.model.idealized_moist_phys import FriersonPhysicsParams, idealized_moist_phys
from jsca.physics.bucket import RAW_BUCKET, ROBERT_BUCKET, bucket_step
from jsca.physics.damping_driver import damping_driver_init
from jsca.physics.mixed_layer import MixedLayerParams

Array = jnp.ndarray


@dataclass(frozen=True)
class BucketModel:
    """A Frierson-style moist model with a land mask and a bucket reservoir.

    ``base`` carries the dynamics + physics (its ``phys`` has the bucket / land
    configuration); ``land`` is the static 0/1 mask; the ``init_bucket_depth*``
    set the reservoir cold start (land finite, ocean effectively infinite so the
    ocean is always "wet").
    """

    base: FriersonModel
    land: Array
    init_bucket_depth_land: float
    init_bucket_depth_ocean: float
    robert_bucket: float
    raw_bucket: float


def build_bucket_model(
    land,
    num_fourier: int = 21,
    nlat: int | None = None,
    nlon: int | None = None,
    dt: float = 720.0,
    num_levels: int = 40,
    # bucket_hydrology vertical coordinate (uneven_sigma; test-case spectral_dynamics_nml)
    scale_heights: float = 11.0,
    surf_res: float = 0.2,
    exponent: float = 7.0,
    robert_coeff: float = 0.03,
    raw_filter_coeff: float = 1.0,
    # mixed_layer_nml (bucket test case)
    mixed_layer_depth: float = 20.0,
    albedo: float = 0.25,
    land_h_capacity_prefactor: float = 0.1,
    land_albedo_prefactor: float = 1.3,
    # surface roughness (ocean values; land_roughness_prefactor defaults to 1)
    roughness: float = 2.0e-4,
    # bucket_nml
    max_bucket_depth_land: float = 2.0,
    init_bucket_depth_land: float = 1.0,
    init_bucket_depth_ocean: float = 1000.0,
    robert_bucket: float = ROBERT_BUCKET,
    raw_bucket: float = RAW_BUCKET,
    **dyn_kwargs,
) -> BucketModel:
    """Build the T``num_fourier`` land + bucket model (grey radiation).

    ``land`` is the ``(nlat, nlon)`` 0/1 mask (south → north), e.g. from
    :func:`jsca.model.land.continents_land_mask` on the model's Gaussian latitudes.
    Defaults reproduce Isca's ``bucket_hydrology`` namelist (with grey radiation and
    ``convection_scheme='SIMPLE_BETTS_MILLER'``).
    """
    nlat = nlat or (2 * num_fourier + 2)
    nlon = nlon or (4 * num_fourier + 4)
    land = jnp.asarray(land, dtype=jnp.float64)
    if land.shape != (nlat, nlon):
        raise ValueError(f"land shape {land.shape} != grid ({nlat}, {nlon})")

    # bucket test case: damping_order=4, uneven_sigma 40-level coordinate.
    dyn_kwargs.setdefault("damping_order", 4)
    dyn = build_dynamics_params(
        num_fourier, nlat, nlon, num_levels,
        vert_coord_option="uneven_sigma", scale_heights=scale_heights,
        surf_res=surf_res, exponent=exponent, **dyn_kwargs)

    p_half_1d, _, p_full_1d, _ = pressure_variables(
        dyn.pk, dyn.bk, jnp.asarray(constants.PSTD_MKS), "simmons_and_burridge")
    phys = FriersonPhysicsParams(
        mixed_layer=MixedLayerParams(depth=mixed_layer_depth, albedo=albedo),
        damping=damping_driver_init(np.asarray(p_full_1d)),
        albedo=albedo,
        roughness_mom=roughness, roughness_heat=roughness, roughness_moist=roughness,
        convection_scheme="SIMPLE_BETTS_MILLER",
        # land / bucket switches
        bucket=True,
        max_bucket_depth_land=max_bucket_depth_land,
        land_h_capacity_prefactor=land_h_capacity_prefactor,
        land_albedo_prefactor=land_albedo_prefactor,
    )

    # A-grid metrics for the tracer horizontal advection (as in build_frierson).
    sin_lat = np.asarray(dyn.transforms.sin_lat)
    sin_edges = np.concatenate([[-1.0], 0.5 * (sin_lat[1:] + sin_lat[:-1]), [1.0]])
    lat_edges = np.arcsin(np.clip(sin_edges, -1.0, 1.0))
    fv = fv_advection_init(nlon, lat_edges, degrees_lon=360.0)

    delta_t = 2.0 * dt
    lat = np.arcsin(sin_lat)[:, None] * np.ones((1, nlon))
    lon = np.linspace(0.0, 2.0 * np.pi, nlon, endpoint=False)[None, :] * np.ones((nlat, 1))
    base = FriersonModel(
        dyn=dyn, phys=phys, fv=fv,
        wave_matrix=build_wave_matrices(dyn.implicit, delta_t),
        wave_matrix_cold=build_wave_matrices(dyn.implicit, dt),
        lat2d=jnp.asarray(lat), lon2d=jnp.asarray(lon), dt=dt, delta_t=delta_t,
        robert_coeff=robert_coeff, raw_filter_coeff=raw_filter_coeff, nlat=nlat, nlon=nlon,
    )
    return BucketModel(
        base=base, land=land,
        init_bucket_depth_land=init_bucket_depth_land,
        init_bucket_depth_ocean=init_bucket_depth_ocean,
        robert_bucket=robert_bucket, raw_bucket=raw_bucket,
    )


def initial_state(m: BucketModel, temperature: float = 264.0, surface_press: float = 1.0e5,
                  tconst: float = 285.0, delta_T: float = 40.0, humidity: float = 2.0e-6,
                  seed: int = 0, perturb: float = 1.0e-4):
    """Cold-start state ``(vors, divs, ts, ln_ps, qg, t_surf, bucket_depth)``.

    The atmosphere/ocean init mirrors :func:`jsca.model.frierson.initial_state`
    (quiescent isothermal atmosphere, meridional SST gradient). The reservoir starts
    at ``init_bucket_depth_land`` over land and ``init_bucket_depth_ocean`` over
    ocean (Isca ``idealized_moist_phys`` L614-617), with both leapfrog levels equal.
    """
    b = m.base
    tf = b.dyn.transforms
    k = b.dyn.num_levels
    rng = np.random.default_rng(seed)
    tg = np.full((k, b.nlat, b.nlon), temperature)
    tg = tg + perturb * rng.standard_normal((k, b.nlat, b.nlon))
    mp = tf.mask_prognostic
    ts = _to_last(grid_to_spectral(tf, jnp.asarray(tg)))
    ts = jnp.where(mp[..., None], ts, 0.0)
    lnps_grid = jnp.full((b.nlat, b.nlon), float(np.log(surface_press)))
    ln_ps = jnp.where(mp, grid_to_spectral(tf, lnps_grid), 0.0)
    zero = jnp.zeros_like(ts)
    qg = jnp.full((b.nlat, b.nlon, k), humidity)
    stack = lambda x: jnp.stack([x, x], axis=-1)  # noqa: E731
    tsurf = tconst - delta_T * ((3.0 * jnp.sin(b.lat2d) ** 2 - 1.0) / 3.0)
    land = m.land > 0.5
    bd0 = jnp.where(land, m.init_bucket_depth_land, m.init_bucket_depth_ocean)
    bucket_depth = jnp.stack([bd0, bd0], axis=-1)
    return (stack(zero), stack(zero), stack(ts), stack(ln_ps), stack(qg), tsurf,
            bucket_depth)


def _step_full(m: BucketModel, state, delta_t: float | None = None, wave_matrix=None):
    """One land + bucket leapfrog step, returning ``(new_state, precip)``.

    Mirrors :func:`jsca.model.frierson._step_full` (identical dynamics, corrections
    and tracer step) with two additions: the column physics receives the land mask
    and the current-level ``bucket_depth`` (so the surface flux takes the bucket
    path and the mixed layer uses land heat capacity/albedo), and after the leapfrog
    the reservoir is advanced by :func:`jsca.physics.bucket.bucket_step`.
    """
    vors, divs, ts, ln_ps, qg, t_surf, bucket_depth = state
    prev, cur, fut = 0, 1, 0
    b = m.base
    dyn, tf = b.dyn, b.dyn.transforms
    dtl = b.delta_t if delta_t is None else delta_t
    wm = b.wave_matrix if wave_matrix is None else wave_matrix

    u_p, v_p, t_p, ps_p = _grid_from_spectral(b, vors, divs, ts, ln_ps, prev)
    _, _, _, ps_c = _grid_from_spectral(b, vors, divs, ts, ln_ps, cur)
    q_p, q_c = qg[..., prev], qg[..., cur]

    ph_p, lph_p, pf_p, lpf_p = pressure_variables(dyn.pk, dyn.bk, ps_p, dyn.vert_difference_option)
    ph_c, lph_c, pf_c, lpf_c = pressure_variables(dyn.pk, dyn.bk, ps_c, dyn.vert_difference_option)
    phi_full, phi_half = compute_geopotential(dyn.pk, t_p, lph_c, lpf_c, dyn.surf_geopotential)
    z_full_c = phi_full / constants.GRAV
    z_half_c = phi_half / constants.GRAV

    gust = b.phys.gust_const * jnp.ones(b.lat2d.shape)
    # land + bucket: pass the static mask and the current-level reservoir depth.
    phys = idealized_moist_phys(
        b.phys, b.lat2d, b.lon2d, u_p, v_p, t_p, q_p, ph_p, pf_p, ph_c, pf_c, z_full_c,
        z_half_c, t_surf, gust, dtl, b.dt,
        land=m.land, bucket_depth=bucket_depth[..., cur])

    mean_sp_prev = area_weighted_global_mean(tf, ps_p)
    energy_p = (0.5 * ((u_p + phys.dt_ug * dtl) ** 2 + (v_p + phys.dt_vg * dtl) ** 2)
                + constants.CP_AIR * (t_p + phys.dt_tg * dtl))
    mean_en_prev = mass_weighted_global_integral(tf, dyn.pk, dyn.bk, energy_p, ps_p)
    mean_water_prev = mass_weighted_global_integral(
        tf, dyn.pk, dyn.bk, q_p + phys.dt_qg * dtl, ps_p)

    dvor, ddiv, dts, dlnps, (wg, ug_c, vg_c, ph_c2) = compute_tendencies(
        dyn, vors, divs, ts, ln_ps, dtl, wm, prev, cur,
        phys.dt_ug, phys.dt_vg, phys.dt_tg, return_diagnostics=True)

    rc, raw = b.robert_coeff, b.raw_filter_coeff
    vors = leapfrog(vors, dvor, prev, cur, fut, dtl, rc, raw)
    divs = leapfrog(divs, ddiv, prev, cur, fut, dtl, rc, raw)
    ts = leapfrog(ts, dts, prev, cur, fut, dtl, rc, raw)
    ln_ps = leapfrog(ln_ps, dlnps, prev, cur, fut, dtl, rc, raw)

    u_f, v_f, t_f, ps_f = _grid_from_spectral(b, vors, divs, ts, ln_ps, fut)
    ps_f2, lnps00_new, _ = mass_correction(tf, ps_f, jnp.real(ln_ps[0, 0, fut]), mean_sp_prev)
    _, ts00_new, _ = energy_correction(
        tf, dyn.pk, dyn.bk, t_f, jnp.real(ts[0, 0, :, fut]), u_f, v_f, ps_f2,
        mean_en_prev, mean_sp_prev)
    ln_ps = ln_ps.at[0, 0, fut].set(lnps00_new + 0.0j)
    ts = ts.at[0, 0, :, fut].set(ts00_new + 0.0j)

    q_cur_new, q_fut, _pf = update_grid_tracer(
        q_p, q_c, phys.dt_qg, ug_c, vg_c, wg, ph_c2, dtl, rc, raw, m.base.fv,
        scheme=FINITE_VOLUME_PARABOLIC)
    q_fut, _ = water_correction(
        tf, dyn.pk, dyn.bk, q_fut, ps_f2, pf_c, mean_water_prev,
        water_correction_limit=200.0e2)

    # --- bucket reservoir: leapfrog + RAW + runoff on the same time indices ---
    bucket_depth = bucket_step(
        bucket_depth, phys.depth_change_cond, phys.depth_change_conv,
        phys.depth_change_lh, m.land > 0.5, prev, cur, fut,
        b.phys.max_bucket_depth_land,
        robert_coeff=m.robert_bucket, raw_filter_coeff=m.raw_bucket)

    roll = lambda a: jnp.stack([a[..., cur], a[..., fut]], axis=-1)  # noqa: E731
    qg = jnp.stack([q_cur_new, q_fut], axis=-1)
    new_state = (roll(vors), roll(divs), roll(ts), roll(ln_ps), qg, phys.t_surf,
                 roll(bucket_depth))
    return new_state, phys.precip


def step(m: BucketModel, state, delta_t: float | None = None, wave_matrix=None):
    """One land + bucket leapfrog step."""
    return _step_full(m, state, delta_t, wave_matrix)[0]


def integrate(m: BucketModel, state, n_steps: int, cold_start: bool = False):
    """Integrate ``n_steps`` with ``lax.scan``; ``cold_start`` runs the first step
    as Isca's forward start-up (``delta_t = dt``)."""
    jstep = jax.jit(lambda s: step(m, s))
    if cold_start and n_steps > 0:
        state = jax.jit(lambda s: step(m, s, m.base.dt, m.base.wave_matrix_cold))(state)
        n_steps -= 1
    state, _ = jax.lax.scan(lambda s, _: (jstep(s), None), state, None, length=n_steps)
    return state


def integrate_climatology(m: BucketModel, state, spinup_steps: int, avg_steps: int,
                          cold_start: bool = True):
    """Spin up then time-average the climatology. Returns ``(state, clim)`` with the
    Frierson fields plus ``bucket_depth`` (current level) in ``clim``."""
    b = m.base
    if cold_start:
        state = jax.jit(lambda s: step(m, s, b.dt, b.wave_matrix_cold))(state)
        spinup_steps = max(spinup_steps - 1, 0)

    jstep = jax.jit(lambda s: step(m, s))
    state, _ = jax.lax.scan(lambda s, _: (jstep(s), None), state, None, length=spinup_steps)

    def diag(s):
        vors, divs, ts, ln_ps, qg, t_surf, bucket_depth = s
        u, v, t, ps = _grid_from_spectral(b, vors, divs, ts, ln_ps, 1)
        return {"ucomp": u, "vcomp": v, "temp": t, "sphum": qg[..., 1],
                "ps": ps, "t_surf": t_surf, "bucket_depth": bucket_depth[..., 1]}

    def body(carry, _):
        s, acc = carry
        s2, precip = _step_full(m, s)
        d = diag(s2)
        d["precip"] = precip
        acc = {kk: acc[kk] + d[kk] for kk in acc}
        return (s2, acc), None

    acc0 = {**{kk: jnp.zeros_like(vv) for kk, vv in diag(state).items()},
            "precip": jnp.zeros(b.lat2d.shape)}
    (state, acc), _ = jax.lax.scan(body, (state, acc0), None, length=avg_steps)
    clim = {kk: np.asarray(vv) / avg_steps for kk, vv in acc.items()}
    return state, clim
