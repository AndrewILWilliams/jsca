"""jsca — JAX port of the Isca idealized GCM framework (Phase 0).

Phase-0 precision policy: everything validates in float64, so importing jsca
enables JAX x64 globally. This must happen before any JAX arrays are created;
import jsca first in scripts. (A float32 production mode is a later phase.)
"""

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.0.1"

# Composable object API (prototype) — a host-side ergonomics layer over the
# functional model core; imported after x64 is enabled (it pulls in
# jax-array-creating modules). `configs` holds the pre-packaged recipes.
from jsca import configs  # noqa: E402, F401
from jsca.api import (  # noqa: E402
    BettsMillerConvection,
    DryForcing,
    GrayRadiation,
    HeldSuarezForcing,
    LargeScaleCondensation,
    Model,
    MoistPhysics,
    RayleighSponge,
    Simulation,
    SpectralDynamics,
    SpectralGrid,
    SurfaceMixedLayer,
)

__all__ = [
    "configs",
    # grid + dynamics
    "SpectralGrid",
    "SpectralDynamics",
    # physics packages
    "MoistPhysics",
    "DryForcing",
    # physics terms
    "GrayRadiation",
    "BettsMillerConvection",
    "LargeScaleCondensation",
    "RayleighSponge",
    "SurfaceMixedLayer",
    "HeldSuarezForcing",
    # assembler + runner
    "Model",
    "Simulation",
]
