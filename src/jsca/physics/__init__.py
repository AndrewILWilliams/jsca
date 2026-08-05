"""Physical parameterizations. Currently the Held–Suarez (1994) idealized forcing."""

from .hs_forcing import (
    HsForcingParams,
    hs_forcing,
    hs_forcing_init,
    newtonian_damping,
    rayleigh_damping,
)

__all__ = [
    "HsForcingParams",
    "hs_forcing_init",
    "hs_forcing",
    "newtonian_damping",
    "rayleigh_damping",
]
