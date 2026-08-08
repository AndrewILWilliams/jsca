"""Pre-packaged recipes: the standard experiment configs, assembled from parts.

Each function returns a :class:`jsca.api.Model` built by combining a
:class:`~jsca.api.SpectralGrid`, :class:`~jsca.api.SpectralDynamics`, and a
physics package from :mod:`jsca.api` — i.e. these are the worked examples of "how
to combine the components to make the Frierson / Held-Suarez config", not bespoke
classes. Copy a body into a notebook cell and tweak it to make a variant.

    import jsca
    model = jsca.configs.frierson(trunc=42)                    # the Frierson config
    model = jsca.configs.frierson(trunc=42,                    # a variant
                                  radiation=jsca.GrayRadiation(solar_constant=1400.0))
    sim = model.initialize()
    sim.run(days=100)
"""
from __future__ import annotations

from jsca.api import (
    DryForcing,
    HeldSuarezForcing,
    Model,
    MoistPhysics,
    SpectralDynamics,
    SpectralGrid,
)


def frierson(trunc: int = 42, dt: float = 720.0, **physics_terms) -> Model:
    """The Frierson (2006) moist aquaplanet config (Isca ``frierson_test_case``).

    Defaults: T``trunc`` on the 25-level pinned sigma coordinate, ``dt`` = 720 s,
    del^8 hyperdiffusion (``damping_order=4``), ``robert_coeff=0.03``. Any
    :class:`~jsca.api.MoistPhysics` term can be overridden as a keyword
    (``radiation=``, ``surface=``, ``sponge=``, ...).
    """
    grid = SpectralGrid(trunc=trunc, dt=dt)  # nlev defaults to the Frierson 25
    dynamics = SpectralDynamics(damping_order=4, robert_coeff=0.03)
    return Model(grid, physics=MoistPhysics(**physics_terms), dynamics=dynamics)


def held_suarez(trunc: int = 21, dt: float = 600.0, num_levels: int = 20,
                forcing: HeldSuarezForcing | None = None) -> Model:
    """The Held & Suarez (1994) dry benchmark config.

    Defaults: T``trunc`` on ``num_levels`` even-sigma levels, ``dt`` = 600 s, del^4
    hyperdiffusion (``damping_order=2``), ``robert_coeff=0.04``. Pass ``forcing``
    to change the Newtonian-relaxation / Rayleigh-friction knobs.
    """
    grid = SpectralGrid(trunc=trunc, dt=dt, nlev=num_levels)
    dynamics = SpectralDynamics(damping_order=2, robert_coeff=0.04)
    phys = DryForcing(forcing=forcing) if forcing is not None else DryForcing()
    return Model(grid, physics=phys, dynamics=dynamics)
