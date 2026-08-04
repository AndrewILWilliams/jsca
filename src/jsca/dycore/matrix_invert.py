"""Small dense matrix inversion for the semi-implicit vertical solves.

Corresponds to Isca's ``src/atmos_spectral/model/matrix_invert.F90``
(``invert``: Gauss–Jordan with column pivoting, returning the inverse in
place plus the determinant).

Documented deviation from the port-the-algorithm rule: the Fortran's 1970s
sequential pivoting loop is replaced by ``jnp.linalg`` (LAPACK/XLA). The
*outputs* are mathematically identical; element-level differences are
round-off-order (the Tier-1 fixture test holds them to ~1e-11 relative for
the nlev-sized, diagonally-dominant matrices the semi-implicit scheme
produces). Porting the literal loop into JAX would be slower, unbatchable,
and no more faithful in its last bits than LAPACK is. The Fortran also
FATALs when ``|det| < 1e-30``; callers should treat a non-finite inverse or
tiny determinant the same way (checked in ``implicit.py`` when it lands).
"""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray


def invert(matrix: Array) -> tuple[Array, Array]:
    """Invert ``(..., n, n)`` matrices; returns ``(inverse, determinant)``."""
    return jnp.linalg.inv(matrix), jnp.linalg.det(matrix)
