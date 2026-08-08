"""Composable object API for building a simulation (prototype).

The philosophy is SpeedyWeather.jl's: **one generic assembler**, not a class per
experiment. You build the pieces —

    grid    -> resolution + coordinates + timestep   (:class:`SpectralGrid`)
    dynamics-> the spectral dynamical-core options    (:class:`SpectralDynamics`)
    physics -> a set of physics *terms*               (:class:`MoistPhysics` /
                                                        :class:`DryForcing`)

— and combine them with :class:`Model`. "The Frierson config" and "the
Held-Suarez config" are then just **recipes** (functions in :mod:`jsca.configs`)
that pick particular pieces, not bespoke classes.

    import jsca
    grid    = jsca.SpectralGrid(trunc=42, dt=720.0)
    physics = jsca.MoistPhysics(radiation=jsca.GrayRadiation(solar_constant=1360.0),
                                surface=jsca.SurfaceMixedLayer(depth=10.0))
    model   = jsca.Model(grid, physics=physics)
    sim     = model.initialize()
    sim.run(days=100)
    sim.state.temperature

This is a **host-side ergonomics layer only** — every method calls the
fixture-validated functional core (``build_frierson`` / ``build_held_suarez`` /
``initial_state`` / ``integrate``). The frozen params pytree the jit'd step sees
is unchanged, so the iron rules hold.

**Honest scope of "composable" (important — read before extending).** Isca's moist
column physics is *not* a bag of independent, reorderable terms: radiation is
split around the surface flux (``gray_rad_down`` feeds the slab ocean, then
``gray_rad_up`` needs ``t_surf``), and the surface / boundary-layer / implicit
vertical-diffusion / mixed-layer block is a single coupled implicit solve
(``vert_diff_down`` -> ``mixed_layer`` -> ``vert_diff_up`` thread one tridiagonal
system). Faithfulness (iron rule 1) means the **call order is fixed by the
driver**. So this API composes *which terms are present and how each is
configured* — it does not (yet) let you arbitrarily reorder or drop terms inside
the coupled moist block. Turning that block into genuinely swappable components
(à la SpeedyWeather) is a real driver refactor, gated by the moist step fixture.

Consequently, of the moist terms, **radiation, the surface/boundary-layer cluster,
and the sponge** thread their parameters through today; **convection and
large-scale condensation are still hardcoded in ``idealized_moist_phys``**, so
their component objects here are structural placeholders (documented on each).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from jsca.model import frierson as _fri
from jsca.model import held_suarez as _hs
from jsca.model.idealized_moist_phys import FriersonPhysicsParams
from jsca.physics.diffusivity import DiffusivityParams
from jsca.physics.hs_forcing import HsForcingParams
from jsca.physics.mixed_layer import MixedLayerParams
from jsca.physics.monin_obukhov import MOParams
from jsca.physics.two_stream_gray_rad import GrayRadParams

_FRIERSON_NLEV = len(_fri.FRIERSON_BK) - 1  # 25

# --- physics-term configs ---------------------------------------------------
# Role-named aliases of the existing frozen physics dataclasses (same hashable,
# jit-static objects the functional core validates against). Aliases, not
# subclasses, so their repr still reads e.g. GrayRadParams(...) — a cosmetic
# rough edge of the prototype.
GrayRadiation = GrayRadParams          # two_stream_gray_rad_nml
HeldSuarezForcing = HsForcingParams    # hs_forcing_nml


@dataclass(frozen=True)
class BettsMillerConvection:
    """Simplified Betts-Miller convection term (``qe_moist_convection``).

    Placeholder config: the port's ``qe_moist_convection`` currently hardcodes the
    Betts-Miller constants (relaxation time, RH target), so this object only marks
    the term's presence. Its knobs will be wired once the driver threads them.
    """


@dataclass(frozen=True)
class LargeScaleCondensation:
    """Large-scale condensation term (``lscale_cond``).

    Placeholder config for the same reason as :class:`BettsMillerConvection`:
    ``idealized_moist_phys`` calls ``lscale_cond`` with its defaults, so ``hc`` /
    ``do_evap`` are not yet threaded. Recorded here for structure.
    """

    hc: float = 1.0
    do_evap: bool = True


@dataclass(frozen=True)
class RayleighSponge:
    """Top-of-model Rayleigh sponge (``damping_driver`` / ``rayleigh_sponge``).

    ``trayfric`` is the lid damping timescale (Isca convention: days if negative,
    seconds if positive); ``pbottom`` the sponge base pressure (Pa). These thread
    through ``build_frierson`` into ``damping_driver_init`` at build time (the
    derived ``nlev_rayfric`` needs the reference pressure profile).
    """

    trayfric: float = -0.25
    pbottom: float = 5000.0
    conserve_energy: bool = True


@dataclass(frozen=True)
class SurfaceMixedLayer:
    """The coupled surface / boundary-layer / slab-ocean block.

    Bundled deliberately: ``surface_flux`` -> ``diffusivity`` -> ``vert_diff_down``
    -> ``mixed_layer`` -> ``vert_diff_up`` form one implicit solve in Isca and
    cannot be split faithfully, so they are configured as a unit. ``depth`` /
    ``albedo`` are the common slab-ocean knobs; the ``monin_obukhov`` and
    ``diffusivity`` sub-configs and the roughness / gustiness scalars expose the
    rest of the block's namelists.
    """

    depth: float = 2.5
    albedo: float = 0.31
    monin_obukhov: MOParams = field(default_factory=MOParams)
    diffusivity: DiffusivityParams = field(default_factory=DiffusivityParams)
    roughness: float = 3.21e-5     # roughness_{mom,heat,moist} (Frierson: equal)
    gust_const: float = 0.0        # constant_gust


# --- grid + dynamics --------------------------------------------------------
@dataclass(frozen=True)
class SpectralGrid:
    """Resolution, coordinates, and timestep (the SpeedyWeather ``SpectralGrid``).

    ``trunc`` is the triangular truncation T``trunc``; ``nlat``/``nlon`` default to
    Isca's Gaussian grid (``2*trunc+2`` x ``4*trunc+4``). ``nlev`` is the number of
    vertical levels (fixed at 25 for the Frierson pinned sigma coordinate; free for
    the dry core). ``dt`` is the physical timestep (s); the leapfrog interval is
    ``2*dt``.
    """

    trunc: int = 42
    dt: float = 720.0
    nlat: int | None = None
    nlon: int | None = None
    nlev: int = _FRIERSON_NLEV
    radius: float | None = None  # None -> constants.RADIUS (Isca's Earth radius)

    @property
    def resolved_nlat(self) -> int:
        return self.nlat if self.nlat is not None else 2 * self.trunc + 2

    @property
    def resolved_nlon(self) -> int:
        return self.nlon if self.nlon is not None else 4 * self.trunc + 4


@dataclass(frozen=True)
class SpectralDynamics:
    """Spectral dynamical-core options (shared by every physics package).

    ``damping_order`` is the hyperdiffusion order (Isca ``damping_order``: 2 -> del^4,
    4 -> del^8). ``robert_coeff``/``raw_filter_coeff`` are the Robert-Asselin / RAW
    time-filter coefficients. ``extra`` passes any other ``build_dynamics_params``
    keyword straight through (e.g. ``alpha_implicit``, ``vert_coord_option``).
    """

    damping_order: int = 4
    robert_coeff: float = 0.03
    raw_filter_coeff: float = 1.0
    extra: dict = field(default_factory=dict)


# --- physics packages -------------------------------------------------------
@dataclass(frozen=True)
class MoistPhysics:
    """The Frierson-family moist column-physics package, composed from its terms.

    All terms default to their standard component, so ``MoistPhysics()`` *is* the
    Frierson stack. Configure a term by passing it (e.g.
    ``radiation=GrayRadiation(solar_constant=1400.0)``). See the module docstring
    for why terms cannot yet be dropped/reordered.
    """

    convection: BettsMillerConvection = field(default_factory=BettsMillerConvection)
    condensation: LargeScaleCondensation = field(default_factory=LargeScaleCondensation)
    radiation: GrayRadParams = field(default_factory=GrayRadParams)
    sponge: RayleighSponge = field(default_factory=RayleighSponge)
    surface: SurfaceMixedLayer = field(default_factory=SurfaceMixedLayer)

    kind = "moist"
    has_moisture = True

    def _physics_params(self) -> FriersonPhysicsParams:
        s = self.surface
        # albedo lives twice in Isca's config (radiation up-pass vs slab ocean);
        # take the surface value as the single source of truth.
        return FriersonPhysicsParams(
            gray_rad=self.radiation,
            mo=s.monin_obukhov,
            diff=s.diffusivity,
            mixed_layer=MixedLayerParams(depth=s.depth, albedo=s.albedo),
            damping=None,  # filled by build_frierson from the reference profile
            roughness_mom=s.roughness, roughness_heat=s.roughness,
            roughness_moist=s.roughness, gust_const=s.gust_const, albedo=s.albedo,
        )

    def _assemble(self, grid: SpectralGrid, dyn: SpectralDynamics):
        if grid.nlev != _FRIERSON_NLEV:
            raise ValueError(
                f"MoistPhysics uses the pinned {_FRIERSON_NLEV}-level Frierson sigma "
                f"coordinate; got nlev={grid.nlev}.")
        params = _fri.build_frierson(
            num_fourier=grid.trunc, nlat=grid.resolved_nlat, nlon=grid.resolved_nlon,
            dt=grid.dt, robert_coeff=dyn.robert_coeff,
            raw_filter_coeff=dyn.raw_filter_coeff, damping_order=dyn.damping_order,
            physics=self._physics_params(),
            sponge_trayfric=self.sponge.trayfric, sponge_pbottom=self.sponge.pbottom,
            sponge_conserve=self.sponge.conserve_energy,
            **_radius_kw(grid), **dyn.extra)
        return params

    def _initial_state(self, params, **ic):
        return _fri.initial_state(params, **ic)

    _integrate = staticmethod(_fri.integrate)
    _grid_from_spectral = staticmethod(_fri._grid_from_spectral)


@dataclass(frozen=True)
class DryForcing:
    """A dry dynamical core with a prescribed forcing (Held-Suarez family).

    The single ``forcing`` term replaces the whole moist column stack with
    Newtonian relaxation + Rayleigh friction. This is the SpeedyWeather
    ``PrimitiveDryModel`` analogue; the prognostic state carries no humidity or
    slab ocean, which is why dry vs moist is the one hard structural fork.
    """

    forcing: HsForcingParams = field(default_factory=HsForcingParams)

    kind = "dry"
    has_moisture = False

    def _assemble(self, grid: SpectralGrid, dyn: SpectralDynamics):
        return _hs.build_held_suarez(
            num_fourier=grid.trunc, nlat=grid.resolved_nlat, nlon=grid.resolved_nlon,
            num_levels=grid.nlev, dt=grid.dt, robert_coeff=dyn.robert_coeff,
            raw_filter_coeff=dyn.raw_filter_coeff, damping_order=dyn.damping_order,
            forcing=self.forcing, **_radius_kw(grid), **dyn.extra)

    def _initial_state(self, params, **ic):
        return _hs.initial_state(params, **ic)

    _integrate = staticmethod(_hs.integrate)
    _grid_from_spectral = staticmethod(_hs._grid_from_spectral)


def _radius_kw(grid: SpectralGrid) -> dict:
    return {} if grid.radius is None else {"radius": grid.radius}


# --- model + simulation -----------------------------------------------------
@dataclass(frozen=True)
class Model:
    """A configured simulation, assembled from a grid, dynamics, and a physics
    package (the SpeedyWeather ``Model`` role). ``initialize()`` freezes the
    configuration into the params pytree and returns a runnable :class:`Simulation`.
    """

    grid: SpectralGrid = field(default_factory=SpectralGrid)
    physics: object = field(default_factory=MoistPhysics)
    dynamics: SpectralDynamics = field(default_factory=SpectralDynamics)

    def initialize(self, **ic_kwargs) -> "Simulation":
        """Build the params pytree and initial condition. ``ic_kwargs`` forward to
        the physics package's ``initial_state`` (e.g. ``humidity``, ``seed``)."""
        params = self.physics._assemble(self.grid, self.dynamics)
        state = self.physics._initial_state(params, **ic_kwargs)
        return Simulation(self, params, state)


class SimulationState:
    """Named, lazy grid-field view of the raw spectral state pytree (current level).

    Returns NumPy arrays. Grid fields are ``(nlat, nlon, K)`` (level-last); surface
    fields ``(nlat, nlon)``. ``sphum`` and ``t_surf`` exist only for moist runs.
    """

    def __init__(self, physics, params, state):
        self._physics = physics
        self._params = params
        self._state = state

    def _grid(self):
        v, d, t, lp = self._state[0], self._state[1], self._state[2], self._state[3]
        return self._physics._grid_from_spectral(self._params, v, d, t, lp, 1)

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
        if not self._physics.has_moisture:
            raise AttributeError("sphum is defined only for moist (MoistPhysics) runs")
        return np.asarray(self._state[4][..., 1])

    @property
    def t_surf(self):
        if not self._physics.has_moisture:
            raise AttributeError("t_surf is defined only for moist (MoistPhysics) runs")
        return np.asarray(self._state[5])

    @property
    def spectral(self):
        """The raw state pytree (for custom diagnostics)."""
        return self._state


class Simulation:
    """A running integration (the SpeedyWeather ``Simulation`` role). Mutable
    host-side driver: :meth:`run` advances ``self._state`` in place and tracks the
    model clock. Heavy work goes through the jit'd functional core.
    """

    def __init__(self, model: Model, params, state):
        self.model = model
        self._params = params
        self._state = state
        self.n_steps = 0  # also gates Isca's cold-start forward step

    @property
    def dt(self) -> float:
        return self._params.dt

    @property
    def day(self) -> float:
        """Model days elapsed since initialization."""
        return self.n_steps * self.dt / 86400.0

    @property
    def state(self) -> SimulationState:
        return SimulationState(self.model.physics, self._params, self._state)

    def run(self, days: float | None = None, steps: int | None = None) -> "Simulation":
        """Advance the simulation. Give exactly one of ``days`` or ``steps``. The
        first ``run`` from the resting initial condition performs Isca's start-up
        forward step automatically (``cold_start``)."""
        if (days is None) == (steps is None):
            raise ValueError("pass exactly one of days= or steps=")
        n = steps if steps is not None else int(round(days * 86400.0 / self.dt))
        if n <= 0:
            return self
        cold_start = self.n_steps == 0
        self._state = self.model.physics._integrate(
            self._params, self._state, n, cold_start=cold_start)
        self.n_steps += n
        return self

    def climatology(self, spinup_days: float, avg_days: float) -> dict:
        """Spin up, then accumulate a time-mean climatology over the averaging
        window (moist runs only). Returns the dict of time-mean grid fields from
        ``integrate_climatology`` and advances the state to the window's end."""
        if not self.model.physics.has_moisture:
            raise NotImplementedError(
                "climatology() is wired for MoistPhysics; the dry core exposes "
                "sampling via integrate(sample_every=...) instead.")
        spinup = int(round(spinup_days * 86400.0 / self.dt))
        avg = int(round(avg_days * 86400.0 / self.dt))
        cold_start = self.n_steps == 0
        self._state, clim = _fri.integrate_climatology(
            self._params, self._state, spinup, avg, cold_start=cold_start)
        self.n_steps += spinup + avg
        return clim
