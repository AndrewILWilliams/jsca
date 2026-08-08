"""A SpeedyWeather.jl-style object API over the functional model core (prototype).

This is a **host-side ergonomics layer only**. It does not add numerics: every
method here calls the already fixture-validated functional core
(:func:`jsca.model.frierson.build_frierson` / ``initial_state`` / ``integrate`` /
``integrate_climatology``). The frozen params pytree that the jit'd step sees is
unchanged, so the iron rules (frozen/hashable config, tables precomputed into
params, pure jit/scan-safe step) are untouched — the objects below are sugar for
configuring and driving a run from a notebook.

Design (mirrors SpeedyWeather.jl's ``SpectralGrid`` -> component structs ->
``Model`` -> ``initialize!`` -> ``run!``):

    import jsca
    grid  = jsca.SpectralGrid(trunc=42, dt=720)          # resolution + timestep
    rad   = jsca.GrayRadiation(solar_constant=1400.0)    # component overrides...
    ocean = jsca.MixedLayer(depth=10.0)
    model = jsca.Frierson(grid, radiation=rad, ocean=ocean)
    sim   = model.initialize()          # build params pytree + initial condition
    sim.run(days=100)                   # advance (jit + lax.scan under the hood)
    sim.state.temperature               # named grid-field accessors (nlat, nlon, K)
    clim = sim.climatology(spinup_days=200, avg_days=100)

The **component structs are the existing frozen physics dataclasses**, re-exported
under role-based names so a notebook user gets tab-completion and docstrings
without hand-assembling ``FriersonPhysicsParams``. They keep their original
``__name__`` (this is a thin alias, not a rename) — that is the one rough edge of
the prototype and would be smoothed in a real implementation.

Scope of this prototype: the **Frierson moist aquaplanet** only (the fuller of the
two assembled models). The Held-Suarez dry benchmark would get the same treatment
(a ``jsca.HeldSuarez`` model class over ``build_held_suarez``). The Frierson
vertical coordinate is the pinned 25-level table, so ``nlev`` is fixed at 25.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from jsca.model.frierson import (
    FRIERSON_BK,
    _grid_from_spectral,
    build_frierson,
    initial_state,
    integrate,
    integrate_climatology,
)
from jsca.model.frierson import (
    FriersonModel as _FriersonParams,
)
from jsca.model.idealized_moist_phys import FriersonPhysicsParams
from jsca.physics.diffusivity import DiffusivityParams
from jsca.physics.mixed_layer import MixedLayerParams
from jsca.physics.monin_obukhov import MOParams
from jsca.physics.two_stream_gray_rad import GrayRadParams

# --- component structs (role-based aliases of the existing frozen dataclasses) ---
# Each is exactly one Isca namelist. Aliasing (not subclassing) keeps them the same
# hashable, jit-static objects the functional core already validates against.
GrayRadiation = GrayRadParams      # two_stream_gray_rad_nml
SurfaceLayer = MOParams            # monin_obukhov_nml (bulk surface fluxes)
BoundaryLayer = DiffusivityParams  # diffusivity_nml (PBL K-profile)
MixedLayer = MixedLayerParams      # mixed_layer_nml (slab ocean)

_FRIERSON_NLEV = len(FRIERSON_BK) - 1  # 25


@dataclass(frozen=True)
class SpectralGrid:
    """Resolution + timestep for a run (the SpeedyWeather ``SpectralGrid`` role).

    ``trunc`` is the triangular truncation (T``trunc``); ``nlat``/``nlon`` default
    to Isca's Gaussian grid for that truncation (``2*trunc+2`` x ``4*trunc+4``).
    ``nlev`` is fixed at the Frierson 25-level pure-sigma coordinate. ``dt`` is the
    physical timestep in seconds (the leapfrog interval is ``2*dt``).
    """

    trunc: int = 42
    dt: float = 720.0
    nlat: int | None = None
    nlon: int | None = None
    nlev: int = _FRIERSON_NLEV
    radius: float | None = None  # None -> Isca's Earth radius (constants.RADIUS)

    @property
    def resolved_nlat(self) -> int:
        return self.nlat if self.nlat is not None else 2 * self.trunc + 2

    @property
    def resolved_nlon(self) -> int:
        return self.nlon if self.nlon is not None else 4 * self.trunc + 4


@dataclass(frozen=True)
class Frierson:
    """A configured Frierson moist aquaplanet (the SpeedyWeather ``Model`` role).

    Bundles the grid with the physics component overrides. Any component left at
    its default uses the Isca Frierson namelist defaults. ``initialize()`` freezes
    this configuration into the params pytree and returns a runnable
    :class:`Simulation`.
    """

    grid: SpectralGrid = field(default_factory=SpectralGrid)
    radiation: GrayRadParams = field(default_factory=GrayRadParams)
    surface_layer: MOParams = field(default_factory=MOParams)
    boundary_layer: DiffusivityParams = field(default_factory=DiffusivityParams)
    ocean: MixedLayerParams = field(default_factory=MixedLayerParams)
    # scalar namelist knobs that live on FriersonPhysicsParams itself
    roughness: float = 3.21e-5     # roughness_{mom,heat,moist} (Frierson: equal)
    gust_const: float = 0.0        # constant_gust
    # dycore / time-filter knobs (forwarded to build_frierson)
    robert_coeff: float = 0.03
    raw_filter_coeff: float = 1.0
    damping_order: int = 4         # Frierson del^8 (see build_frierson docstring)

    def _build_physics(self) -> FriersonPhysicsParams:
        # albedo is stored twice in Isca's config (radiation up-pass reads
        # FriersonPhysicsParams.albedo; the slab ocean reads mixed_layer.albedo).
        # Keep them consistent by taking the ocean's value as the single source.
        return FriersonPhysicsParams(
            gray_rad=self.radiation,
            mo=self.surface_layer,
            diff=self.boundary_layer,
            mixed_layer=self.ocean,
            damping=None,  # filled by build_frierson from the reference profile
            roughness_mom=self.roughness,
            roughness_heat=self.roughness,
            roughness_moist=self.roughness,
            gust_const=self.gust_const,
            albedo=self.ocean.albedo,
        )

    def initialize(self, **ic_kwargs) -> "Simulation":
        """Build the params pytree and the Isca-matched initial condition.

        ``ic_kwargs`` are forwarded to
        :func:`jsca.model.frierson.initial_state` (e.g. ``humidity``, ``seed``,
        ``delta_T``, ``perturb``).
        """
        g = self.grid
        if g.nlev != _FRIERSON_NLEV:
            raise ValueError(
                f"Frierson uses the pinned {_FRIERSON_NLEV}-level sigma coordinate; "
                f"got nlev={g.nlev}.")
        dyn_kwargs = {}
        if g.radius is not None:
            dyn_kwargs["radius"] = g.radius
        params = build_frierson(
            num_fourier=g.trunc,
            nlat=g.resolved_nlat,
            nlon=g.resolved_nlon,
            dt=g.dt,
            robert_coeff=self.robert_coeff,
            raw_filter_coeff=self.raw_filter_coeff,
            damping_order=self.damping_order,
            physics=self._build_physics(),
            **dyn_kwargs,
        )
        state = initial_state(params, **ic_kwargs)
        return Simulation(model=self, _params=params, _state=state)


class SimulationState:
    """Named, lazy grid-field view of the raw spectral state tuple.

    The functional core keeps state as an anonymous pytree
    ``(vors, divs, ts, ln_ps, qg, t_surf)``; this view transforms it to the
    physical grid fields on access (current time level), returning NumPy arrays.
    Grid fields are ``(nlat, nlon, K)`` (level-last); surface fields ``(nlat, nlon)``.
    """

    def __init__(self, params: _FriersonParams, state):
        self._params = params
        self._state = state

    def _grid(self):
        vors, divs, ts, ln_ps = self._state[0], self._state[1], self._state[2], self._state[3]
        return _grid_from_spectral(self._params, vors, divs, ts, ln_ps, 1)  # current level

    @property
    def u(self):
        return np.asarray(self._grid()[0])

    @property
    def v(self):
        return np.asarray(self._grid()[1])

    @property
    def temperature(self):
        return np.asarray(self._grid()[2])

    @property
    def surface_pressure(self):
        return np.asarray(self._grid()[3])

    @property
    def sphum(self):
        return np.asarray(self._state[4][..., 1])  # current-level humidity tracer

    @property
    def t_surf(self):
        return np.asarray(self._state[5])

    @property
    def spectral(self):
        """The raw ``(vors, divs, ts, ln_ps, qg, t_surf)`` pytree (for custom diagnostics)."""
        return self._state


class Simulation:
    """A running Frierson integration (the SpeedyWeather ``Simulation`` role).

    Mutable host-side driver: :meth:`run` advances ``self._state`` in place (like
    SpeedyWeather's ``run!``) and tracks the model clock. All heavy work goes
    through the jit'd functional core.
    """

    def __init__(self, model: Frierson, _params: _FriersonParams, _state):
        self.model = model
        self._params = _params
        self._state = _state
        self.n_steps = 0  # steps taken; also gates Isca's cold-start forward step

    @property
    def dt(self) -> float:
        return self._params.dt

    @property
    def day(self) -> float:
        """Model days elapsed since initialization."""
        return self.n_steps * self.dt / 86400.0

    @property
    def state(self) -> SimulationState:
        return SimulationState(self._params, self._state)

    def run(self, days: float | None = None, steps: int | None = None) -> "Simulation":
        """Advance the simulation. Give exactly one of ``days`` or ``steps``.

        The first ``run`` from the resting initial condition performs Isca's
        start-up forward step automatically (``cold_start``).
        """
        if (days is None) == (steps is None):
            raise ValueError("pass exactly one of days= or steps=")
        n = steps if steps is not None else int(round(days * 86400.0 / self.dt))
        if n <= 0:
            return self
        cold_start = self.n_steps == 0
        self._state = integrate(self._params, self._state, n, cold_start=cold_start)
        self.n_steps += n
        return self

    def climatology(self, spinup_days: float, avg_days: float) -> dict:
        """Spin up then accumulate a time-mean climatology over the averaging window.

        Returns the dict of time-mean grid fields from
        :func:`jsca.model.frierson.integrate_climatology` (``ucomp``, ``vcomp``,
        ``temp``, ``sphum``, ``ps``, ``t_surf``, ``precip``). Advances the
        simulation state to the end of the window and counts the clock.
        """
        spinup = int(round(spinup_days * 86400.0 / self.dt))
        avg = int(round(avg_days * 86400.0 / self.dt))
        cold_start = self.n_steps == 0
        self._state, clim = integrate_climatology(
            self._params, self._state, spinup, avg, cold_start=cold_start)
        self.n_steps += spinup + avg
        return clim
