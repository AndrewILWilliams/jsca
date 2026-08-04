"""Gaussian latitudes and quadrature weights.

Faithful port of ``compute_gaussian`` from Isca's
``src/atmos_spectral/tools/gauss_and_legendre.F90`` (GFDL FMS, GPL-3.0):
Newton iteration on the Legendre polynomial P_n (Numerical Recipes ``gauleg``),
hemisphere-only, with convergence ``0.1**precision`` (1e-15 in float64) and
``itermax = 10``.

Conventions
-----------
* ``gaussian_hemisphere(n_hem)`` reproduces the Fortran routine exactly:
  ``sin_hem[0]`` is the node closest to the pole (largest mu), descending.
* ``gaussian_grid(nlat)`` assembles the global grid **south → north**
  (mu ascending), which is the jsca-wide latitude ordering. If Fortran
  fixture comparisons later reveal Isca assembles the opposite way, flip at
  the fixture boundary — not here.

The independent cross-check against ``numpy.polynomial.legendre.leggauss``
lives in the test suite, keeping the runtime path faithful to the Fortran.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

_CONVERG = 0.1**15  # Fortran: converg = .1**precision(real8) = 1e-15
_ITERMAX = 10


class GaussianGrid(NamedTuple):
    """Global Gaussian grid, south → north (mu ascending)."""

    sin_lat: np.ndarray  # mu = sin(latitude), shape (nlat,)
    wts_lat: np.ndarray  # quadrature weights, sum = 2, shape (nlat,)
    lat: np.ndarray  # latitude in radians, shape (nlat,)


def gaussian_hemisphere(n_hem: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact port of ``compute_gaussian(sin_hem, wts_hem, n_hem)``.

    Returns the ``n_hem`` positive Gaussian nodes (descending from the pole)
    and their weights for a global grid of ``2 * n_hem`` latitudes.
    """
    n = 2 * n_hem
    sin_hem = np.empty(n_hem, dtype=np.float64)
    wts_hem = np.empty(n_hem, dtype=np.float64)
    for i in range(1, n_hem + 1):
        z = np.cos(np.pi * (i - 0.25) / (n + 0.5))
        pp = np.nan
        for _ in range(_ITERMAX):
            p1, p2 = 1.0, 0.0
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) < _CONVERG:
                break
        else:
            raise RuntimeError(
                "gaussian_hemisphere: abscissas failed to converge "
                f"in {_ITERMAX} iterations (n_hem={n_hem})"
            )
        sin_hem[i - 1] = z
        wts_hem[i - 1] = 2.0 / ((1.0 - z * z) * pp * pp)
    return sin_hem, wts_hem


def gaussian_grid(nlat: int) -> GaussianGrid:
    """Global Gaussian grid with ``nlat`` latitudes (must be even), south → north."""
    if nlat % 2 != 0:
        raise ValueError(f"nlat must be even, got {nlat}")
    sin_hem, wts_hem = gaussian_hemisphere(nlat // 2)
    # sin_hem is descending positive; global south→north is [-mu_max … -mu_min, mu_min … mu_max]
    sin_lat = np.concatenate([-sin_hem, sin_hem[::-1]])
    wts_lat = np.concatenate([wts_hem, wts_hem[::-1]])
    return GaussianGrid(sin_lat=sin_lat, wts_lat=wts_lat, lat=np.arcsin(sin_lat))
