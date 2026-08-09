"""Isca's single-column model (SCM): the physics stack with the dynamics bypassed.

Faithful port of Isca's ``src/atmos_column`` driver — ``column.F90`` (the time
loop), ``column_grid.F90`` (the column grid), ``column_init_cond.F90`` /
``column_initialize_fields.F90`` (the cold-start initial condition) — wired to the
already-assembled Frierson column physics
(:func:`jsca.model.idealized_moist_phys.idealized_moist_phys`).

Why this exists (scoping doc Sec 4.4, Tier 2): the SCM runs the full physics *chain*
(convection -> condensation -> radiation -> surface flux -> boundary-layer
diffusion -> slab ocean) over many steps **without any dynamical core**, so a
jsca-vs-Isca comparison isolates the physics from spectral-dynamics chaos. It is
also the natural home for testing new convection/radiation schemes, and it runs
orders of magnitude faster than the full model. Cite McKim et al. (2024) for the
Isca SCM.

## What the single-column model actually does (``column.F90``)

The SCM is a **drop-in replacement for ``spectral_dynamics``** in the solo driver
(``atmosphere.F90`` L319-323, ``#ifdef COLUMN_MODEL``). Where the spectral core
computes dynamical tendencies and does the semi-implicit solve, the column does
**none of that**. Each step it simply:

* advances **temperature** with the grid-space leapfrog+RAW filter using the
  physics tendency ``dt_tg`` (``column.F90`` L271, ``leapfrog_3d_real``);
* advances the **tracers** (humidity) the same way, with an optional
  ``q_decrease_only`` stratospheric clamp (``column.F90`` L344-346);
* holds the **winds and surface pressure fixed** at their initial values — note
  ``column`` returns ``ug/vg/psg`` from the *previous* slot unchanged
  (``column.F90`` L358-360). The momentum tendency ``dt_ug/dt_vg`` from the
  boundary-layer diffusion is **computed and discarded**: the implicit assumption
  is that "the dynamics" would restore the prescribed surface wind, so d(u)/dt = 0
  (see ``exp/test_cases/column_test_case/column_test.py`` header).

The slab-ocean ``t_surf`` still evolves — it is updated *inside* the physics
(``mixed_layer``), exactly as in the full model.

Because surface pressure is fixed, ``p_half``/``p_full`` are constant in time and
the "previous" and "current" pressures Isca's physics distinguishes are identical
here; only the geopotential heights change (they follow the evolving temperature).

## Layout / conventions (CLAUDE.md rule 5)

Grid columns are level-last ``(nlat, nlon, K)`` (k = 0 top … K-1 surface); the two
leapfrog time levels are the **last** axis of ``tg``/``qg`` (``(nlat, nlon, K, 2)``)
so :func:`jsca.dycore.leapfrog.leapfrog` acts on ``a[..., slot]`` and a slice
``tg[..., slot]`` is a clean level-last column set for the physics. Winds and
surface pressure carry no time axis (they are constant).

## Validation status (CLAUDE.md rules 2-3)

Every physics kernel composed here is golden-fixture-validated against the real
Fortran, and :func:`jsca.dycore.leapfrog.leapfrog` is the exact port of
``leapfrog_3d_real`` (verified line-by-line — the SCM's real-tracer variant applies
``raw_filter_coeff`` on the current-level update, unlike the dycore's
``leapfrog_2level_A_3d_real`` quirk). The genuinely new numerics are the cold-start
initial condition (:func:`initial_state`, ``column_initialize_fields.F90``) and the
``q_decrease_only`` clamp (:func:`jsca.dycore.leapfrog.apply_q_decrease_only`) —
both simple arithmetic. A machine-precision golden **column-step** fixture from an
instrumented Isca SCM run is the remaining validation and needs a full Isca build
(driver stub: ``fortran_instrumentation/dump_column_init_reference.F90``); until
then the assembly is gated by the stability/conservation smoke test in
``tests/test_column.py`` (the accepted pattern for jsca's assembly modules —
cf. ``tests/test_frierson.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jsca import constants
from jsca.dycore.leapfrog import apply_q_decrease_only, leapfrog
from jsca.dycore.press_and_geopot import compute_geopotential, pressure_variables
from jsca.dycore.vert_coordinate import compute_vert_coord
from jsca.model.idealized_moist_phys import (
    FriersonPhysicsParams,
    idealized_moist_phys,
)
from jsca.physics.damping_driver import damping_driver_init
from jsca.physics.mixed_layer import MixedLayerParams

Array = jnp.ndarray

# The 25-level pure-sigma coordinate used by the canonical column test case
# (exp/test_cases/column_test_case/column_test.py vert_coordinate_nml) -- the same
# coordinate as the Frierson aquaplanet, so the SCM exercises the physics at the
# levels they were validated on.
COLUMN_BK = np.array([
    0.000000, 0.0117665, 0.0196679, 0.0315244, 0.0485411, 0.0719344, 0.1027829,
    0.1418581, 0.1894648, 0.2453219, 0.3085103, 0.3775033, 0.4502789, 0.5244989,
    0.5977253, 0.6676441, 0.7322627, 0.7900587, 0.8400683, 0.8819111, 0.9157609,
    0.9422770, 0.9625127, 0.9778177, 0.9897489, 1.0000000])
COLUMN_PK = np.zeros(26)

# Frierson p2-insolation "global-average" latitude: 3 sin^2(lat) - 1 = 0, so the
# lat at which the Frierson-2006 shortwave profile equals the global mean S/4.
# exp/test_cases/column_test_case/column_test.py sets lat_value to this.
GLOBAL_AVERAGE_LAT_DEG = float(np.rad2deg(np.arcsin(1.0 / np.sqrt(3.0))))


@dataclass(frozen=True)
class ColumnModel:
    """Static configuration for the single-column model.

    ``pk``/``bk`` are the vertical-coordinate interface coefficients; ``lat2d``/
    ``lon2d`` are the column centre coordinates in radians (shape ``(nlat, nlon)``);
    winds and surface pressure never change so they live in the initial state, not
    here. ``robert_coeff``/``raw_filter_coeff`` default to Isca's ``column_nml``
    values (0.0 / 1.0) -- with no dynamics there is no gravity-wave noise for the
    time filter to control, so the SCM leaves it off.
    """

    phys: FriersonPhysicsParams
    pk: Array
    bk: Array
    lat2d: Array
    lon2d: Array
    surf_geopotential: Array
    num_levels: int
    nlat: int
    nlon: int
    dt: float
    delta_t: float
    robert_coeff: float
    tracer_robert_coeff: float
    raw_filter_coeff: float
    q_decrease_only: bool
    vert_difference_option: str


def build_column(
    lat_value: float = GLOBAL_AVERAGE_LAT_DEG,
    longitude: float = 0.0,
    latitudes: np.ndarray | None = None,
    longitudes: np.ndarray | None = None,
    dt: float = 1440.0,                     # column_test main_nml dt_atmos
    num_levels: int | None = None,
    pk: np.ndarray | None = None,
    bk: np.ndarray | None = None,
    vert_coord_option: str = "even_sigma",  # column_nml default
    surf_geopotential: float = 0.0,         # column_init_cond_nml default
    robert_coeff: float = 0.0,              # column_nml default
    tracer_robert_coeff: float | None = None,
    raw_filter_coeff: float = 1.0,          # column_nml default
    q_decrease_only: bool = True,           # column_test.py
    mixed_layer_depth: float = 2.5,         # column_test mixed_layer_nml
    albedo: float = 0.30,                   # column_test mixed_layer_nml albedo_value
    do_evap: bool = False,                  # column_test lscale_cond_nml do_evap
    vert_difference_option: str = "simmons_and_burridge",
    **phys_kwargs,
) -> ColumnModel:
    """Build a single-column (or a small lat/lon set of independent columns) model.

    The grid is a set of **independent** columns — there is no horizontal coupling,
    so lat/lon only enter the physics through insolation and the surface state.

    * A single column at ``lat_value`` degrees (default: the Frierson global-average
      latitude, ``exp/.../column_test.py``) and ``longitude`` degrees.
    * Or an arbitrary set via ``latitudes``/``longitudes`` (degrees, 1-D each); the
      grid is their outer product.

    Vertical coordinate: pass explicit ``pk``/``bk`` (default: the 25-level Frierson
    :data:`COLUMN_BK`), or a ``vert_coord_option`` + ``num_levels`` for
    :func:`jsca.dycore.vert_coordinate.compute_vert_coord`.

    **Faithfulness note (ambiguity surfaced, CLAUDE.md).** The canonical Isca column
    namelist sets *both* ``column_nml:num_levels = 31`` (with the default
    ``vert_coord_option = 'even_sigma'``) *and* an explicit 25-level
    ``vert_coordinate_nml``; since ``even_sigma`` does not read that namelist, Isca
    would actually run 31 even-sigma levels there. jsca does not replicate that
    latent inconsistency: it uses whichever coordinate you pass explicitly (default
    the 25-level Frierson one, so the physics runs on validated levels). Pass
    ``vert_coord_option='even_sigma', num_levels=31, pk=None, bk=None`` to reproduce
    the levels Isca's default column run really uses.
    """
    if pk is None or bk is None:
        if num_levels is None:
            pk = COLUMN_PK if pk is None else np.asarray(pk)
            bk = COLUMN_BK if bk is None else np.asarray(bk)
        else:
            pk, bk = compute_vert_coord(vert_coord_option, num_levels)
    pk = np.asarray(pk, dtype=float)
    bk = np.asarray(bk, dtype=float)
    nlev = len(bk) - 1
    if num_levels is not None and num_levels != nlev:
        raise ValueError(
            f"num_levels={num_levels} is inconsistent with the {nlev}-level pk/bk")

    lat_deg = np.asarray([lat_value] if latitudes is None else latitudes, dtype=float)
    lon_deg = np.asarray([longitude] if longitudes is None else longitudes, dtype=float)
    nlat, nlon = lat_deg.size, lon_deg.size
    lat2d = np.deg2rad(lat_deg)[:, None] * np.ones((1, nlon))
    lon2d = np.deg2rad(lon_deg)[None, :] * np.ones((nlat, 1))

    # reference full-level pressures for the sponge depth (unused when do_damping is
    # off, as in the SCM, but damping_driver_init needs a reference profile).
    _, _, p_full_1d, _ = pressure_variables(
        pk, bk, jnp.asarray(constants.PSTD_MKS), vert_difference_option)
    phys = FriersonPhysicsParams(
        mixed_layer=MixedLayerParams(depth=mixed_layer_depth, albedo=albedo),
        damping=damping_driver_init(np.asarray(p_full_1d)),
        albedo=albedo,
        do_evap=do_evap,
        **phys_kwargs,
    )

    return ColumnModel(
        phys=phys,
        # pk/bk stay NumPy (static): pressure_variables branches on bool(pk[0]==0)
        # and compute_geopotential on bool(pk[0]==0), which must be concrete under jit.
        pk=pk, bk=bk,
        lat2d=jnp.asarray(lat2d), lon2d=jnp.asarray(lon2d),
        surf_geopotential=jnp.full((nlat, nlon), float(surf_geopotential)),
        num_levels=nlev, nlat=nlat, nlon=nlon,
        dt=dt, delta_t=2.0 * dt,
        robert_coeff=robert_coeff,
        tracer_robert_coeff=robert_coeff if tracer_robert_coeff is None
        else tracer_robert_coeff,
        raw_filter_coeff=raw_filter_coeff,
        q_decrease_only=q_decrease_only,
        vert_difference_option=vert_difference_option,
    )


def initial_state(
    m: ColumnModel,
    initial_temperature: float = 264.0,     # column_init_cond_nml
    reference_sea_level_press: float = 101325.0,  # column_nml
    surface_wind: float = 5.0,              # column_init_cond_nml
    initial_sphum: float = 1.0e-3,          # column_test column_nml
    t_surf: float | None = None,            # default: mixed_layer tconst
) -> tuple:
    """Isca's SCM cold-start initial condition — port of ``column_initialize_fields``
    (``column_initialize_fields.F90`` L43-89) and ``column_init_cond`` (surface
    geopotential + tracers).

    Returns ``(u, v, tg, qg, ps, t_surf)``:

    * **winds** ``u = v = surface_wind / sqrt(2)`` at the *bottom* level, zero
      aloft (L74-75), so ``sqrt(u^2 + v^2) = surface_wind`` at the surface — the
      prescribed wind that drives the surface fluxes. Held fixed thereafter.
    * **temperature** uniform ``initial_temperature`` (L77); ``tg``/``qg`` carry
      two identical time levels (cold start: previous == current, ``column.F90``
      L702-713).
    * **surface pressure** from hydrostatic balance with the surface geopotential
      (L78-79): ``ps = exp(ln(p_ref) - Phi_s / (Rd * T0))``. Held fixed.
    * **humidity** uniform ``initial_sphum`` (``column.F90`` L694). Isca's SCM
      seeds ``sphum`` uniformly rather than the aquaplanet's 2e-6.
    * **slab SST** ``t_surf``: Isca initialises the slab (``mixed_layer_bc`` case,
      no restart) as ``t_surf = t_surf_init + 1.0`` where ``t_surf_init`` is the
      lowest model-level temperature (``idealized_moist_phys.F90`` L643) -- a
      **deliberately +1 K unstable** start "to allow moisture to quickly enter the
      atmosphere avoiding problems with the convection scheme" (Isca's comment). So
      the default here is ``initial_temperature + 1.0``. This +1 K is the single
      largest jsca-vs-Isca difference if omitted: it offsets the entire SST
      trajectory by ~1 K early (decaying to ~0.3 K by day 40). Verified against a
      real Isca column run -- with the +1 K the day-40 SST RMSD drops from 0.55 K
      to 0.02 K. Pass ``t_surf`` explicitly to override. Passed separately because
      the ocean is the surface's state, not the column's.
    """
    k, nlat, nlon = m.num_levels, m.nlat, m.nlon
    u = np.zeros((nlat, nlon, k))
    v = np.zeros((nlat, nlon, k))
    u[..., -1] = surface_wind / np.sqrt(2.0)   # bottom level = surface (k = K-1)
    v[..., -1] = surface_wind / np.sqrt(2.0)

    t_col = np.full((nlat, nlon, k), float(initial_temperature))
    ln_ps = np.log(reference_sea_level_press) - np.asarray(m.surf_geopotential) / (
        constants.RDGAS * initial_temperature)
    ps = np.exp(ln_ps)
    q_col = np.full((nlat, nlon, k), float(initial_sphum))

    stack = lambda a: jnp.stack([jnp.asarray(a), jnp.asarray(a)], axis=-1)  # noqa: E731
    # Isca's deliberately-unstable slab init: lowest-level temp + 1 K
    # (idealized_moist_phys.F90 L643). Override via t_surf for a prescribed SST.
    tsurf = float(initial_temperature) + 1.0 if t_surf is None else t_surf
    return (
        jnp.asarray(u), jnp.asarray(v),
        stack(t_col), stack(q_col), jnp.asarray(ps),
        jnp.full((nlat, nlon), float(tsurf)),
    )


def _step_full(m: ColumnModel, state, delta_t: float | None = None):
    """One SCM step, returning ``(new_state, precip)``. ``step`` drops ``precip``.

    Reproduces the ``atmosphere.F90`` COLUMN_MODEL flow (L300-323): physics on the
    previous level with the current geopotential heights, then the grid-space
    leapfrog of T and q only; winds and surface pressure are untouched.
    """
    u, v, tg, qg, ps, t_surf = state
    prev, cur, fut = 0, 1, 0                 # slot roll, identical to frierson.py
    dtl = m.delta_t if delta_t is None else delta_t

    t_prev, t_cur, q_prev = tg[..., prev], tg[..., cur], qg[..., prev]

    # Surface pressure is fixed, so previous and current pressures coincide.
    ph, lph, pf, lpf = pressure_variables(m.pk, m.bk, ps, m.vert_difference_option)
    # Geopotential heights from the CURRENT-level temperature: atmosphere.F90
    # recomputes z_full/z_half(current) from tg(future -> current) at the end of the
    # prior step and hands *those* current heights to the physics (L335-337, L301).
    phi_full, phi_half = compute_geopotential(m.pk, t_cur, lph, lpf, m.surf_geopotential)
    z_full = phi_full / constants.GRAV
    z_half = phi_half / constants.GRAV

    # Frierson sets constant_gust = 0, so the steady gustiness the diffusivity path
    # sees is 0 (as in frierson.py -- the cold-start transient 1.0 m/s is not
    # reproduced, a documented negligible start-up difference).
    gust = m.phys.gust_const * jnp.ones(m.lat2d.shape)
    phys = idealized_moist_phys(
        m.phys, m.lat2d, m.lon2d, u, v, t_prev, q_prev,
        ph, pf, ph, pf, z_full, z_half, t_surf, gust, dtl, m.dt)

    rc, traw, raw = m.robert_coeff, m.tracer_robert_coeff, m.raw_filter_coeff
    tg = leapfrog(tg, phys.dt_tg, prev, cur, fut, dtl, rc, raw)
    # Momentum tendency dt_ug/dt_vg is discarded (winds prescribed) -- column.F90
    # never leapfrogs ug/vg. Humidity uses the tracer's robert_coeff.
    qg = leapfrog(qg, phys.dt_qg, prev, cur, fut, dtl, traw, raw)
    if m.q_decrease_only:
        qg = qg.at[..., fut].set(apply_q_decrease_only(qg[..., fut]))

    roll = lambda a: jnp.stack([a[..., cur], a[..., fut]], axis=-1)  # noqa: E731
    new_state = (u, v, roll(tg), roll(qg), ps, phys.t_surf)
    return new_state, phys.precip


def step(m: ColumnModel, state, delta_t: float | None = None):
    """One single-column step. ``state = (u, v, tg, qg, ps, t_surf)``."""
    return _step_full(m, state, delta_t)[0]


def integrate(m: ColumnModel, state, n_steps: int, cold_start: bool = False):
    """Integrate ``n_steps`` with ``lax.scan``. ``cold_start`` runs the first step
    as Isca's forward start-up step (``delta_t = dt``, ``column.F90`` L259-263)."""
    jstep = jax.jit(lambda s: step(m, s))
    if cold_start and n_steps > 0:
        state = jax.jit(lambda s: step(m, s, m.dt))(state)
        n_steps -= 1
    state, _ = jax.lax.scan(lambda s, _: (jstep(s), None), state, None, length=n_steps)
    return state


def integrate_climatology(m: ColumnModel, state, spinup_steps: int, avg_steps: int,
                          cold_start: bool = True):
    """Spin up, then accumulate the time-mean column climatology over the averaging
    window. Returns ``(state, clim)`` with time-mean fields: ``temp``/``sphum``
    ``(nlat, nlon, K)`` at the current level, ``t_surf`` ``(nlat, nlon)``, and
    ``precip`` ``(nlat, nlon)`` (kg/m^2/s). Everything runs inside ``lax.scan``."""
    if cold_start:
        state = jax.jit(lambda s: step(m, s, m.dt))(state)
        spinup_steps = max(spinup_steps - 1, 0)

    jstep = jax.jit(lambda s: step(m, s))
    state, _ = jax.lax.scan(lambda s, _: (jstep(s), None), state, None, length=spinup_steps)

    def diag(s):
        u, v, tg, qg, ps, t_surf = s
        return {"temp": tg[..., 1], "sphum": qg[..., 1], "t_surf": t_surf}

    def body(carry, _):
        s, acc = carry
        s2, precip = _step_full(m, s)
        d = diag(s2)
        d["precip"] = precip
        acc = {kk: acc[kk] + d[kk] for kk in acc}
        return (s2, acc), None

    acc0 = {**{kk: jnp.zeros_like(vv) for kk, vv in diag(state).items()},
            "precip": jnp.zeros(m.lat2d.shape)}
    (state, acc), _ = jax.lax.scan(body, (state, acc0), None, length=avg_steps)
    clim = {kk: np.asarray(vv) / avg_steps for kk, vv in acc.items()}
    return state, clim
