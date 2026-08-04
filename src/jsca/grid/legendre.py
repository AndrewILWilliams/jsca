"""Normalized associated Legendre functions — exact port of ``compute_legendre``
from Isca's ``src/atmos_spectral/tools/gauss_and_legendre.F90``.

Normalization (as inherited from the Fortran; do not "fix"):
    integral_{-1}^{1} [P~_l^m(mu)]^2 dmu = 1,
i.e. ``P~_0^0 = sqrt(1/2)`` (the Fortran's ``poly(0,0) = sqrt(0.5)``), with **no
Condon–Shortley phase** (the recursion generates all-positive sectoral values).
Relation to the 4pi/geodesy-normalized ALFs: ``P~ = P^{4pi} / sqrt(2)``.

Storage layout follows the GFDL convention: index ``(m, n)`` with total
wavenumber ``l = m + n`` (see ``jsca.grid.spectral``). Output is arranged
``(nlat, m, n)`` so spectral synthesis/analysis are single einsums over
``(m, n)`` blocks.

Recursion (identical to the Fortran, vectorized over m and latitude):
    eps(m, n)   = sqrt(((m+n)^2 - m^2) / (4 (m+n)^2 - 1))
    P~(0, 0)    = sqrt(0.5)
    P~(m, 0)    = sqrt(0.5 (2m+1)/m) * cos(theta) * P~(m-1, 0)
    P~(m, 1)    = mu * P~(m, 0) / eps(m, 1)
    P~(m, n)    = (mu * P~(m, n-1) - eps(m, n-1) * P~(m, n-2)) / eps(m, n)
"""

from __future__ import annotations

import numpy as np

from .spectral import Truncation


def epsilon(trunc: Truncation) -> np.ndarray:
    """eps(m, n) on the storage grid, shape ``trunc.shape`` (float64)."""
    fourier_max = trunc.num_fourier * trunc.fourier_inc
    m = np.arange(fourier_max + 1, dtype=np.float64)[:, None]
    n = np.arange(trunc.num_spherical + 1, dtype=np.float64)[None, :]
    m2 = m**2
    l2 = (m + n) ** 2
    return np.sqrt((l2 - m2) / (4.0 * l2 - 1.0))


def compute_legendre(trunc: Truncation, sin_lat: np.ndarray) -> np.ndarray:
    """Normalized ALFs, shape ``(nlat, num_fourier+1, num_spherical+1)``.

    Exact port of the Fortran ``compute_legendre`` (vectorized; identical
    operation order per (m, n), so values match to the last ulp modulo
    non-associative summation — there is none here, it's a pure recursion).
    """
    sin_lat = np.asarray(sin_lat, dtype=np.float64)
    (nlat,) = sin_lat.shape
    fourier_max = trunc.num_fourier * trunc.fourier_inc
    n_sph = trunc.num_spherical
    cos_lat = np.sqrt(1.0 - sin_lat * sin_lat)  # as in the Fortran

    eps = epsilon(trunc)  # (fourier_max+1, n_sph+1)
    b = np.zeros(fourier_max + 1)
    m_arr = np.arange(1, fourier_max + 1, dtype=np.float64)
    b[1:] = np.sqrt(0.5 * (2.0 * m_arr + 1.0) / m_arr)

    poly = np.zeros((nlat, fourier_max + 1, n_sph + 1), dtype=np.float64)
    # n = 0 diagonal (sectoral, l = m): cumulative product over m
    poly[:, 0, 0] = np.sqrt(0.5)
    for m in range(1, fourier_max + 1):
        poly[:, m, 0] = b[m] * cos_lat * poly[:, m - 1, 0]
    # n = 1
    poly[:, :, 1] = sin_lat[:, None] * poly[:, :, 0] / eps[None, :, 1]
    # n >= 2
    for n in range(2, n_sph + 1):
        poly[:, :, n] = (
            sin_lat[:, None] * poly[:, :, n - 1] - eps[None, :, n - 1] * poly[:, :, n - 2]
        ) / eps[None, :, n]

    if trunc.fourier_inc == 1:
        return poly
    return poly[:, :: trunc.fourier_inc, :][:, : trunc.num_fourier + 1, :]
