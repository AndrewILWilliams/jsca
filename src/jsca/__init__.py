"""jsca — JAX port of the Isca idealized GCM framework (Phase 0).

Phase-0 precision policy: everything validates in float64, so importing jsca
enables JAX x64 globally. This must happen before any JAX arrays are created;
import jsca first in scripts. (A float32 production mode is a later phase.)
"""

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.0.1"
