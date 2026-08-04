"""Spectral truncation layout, matching the GFDL spectral core's storage.

GFDL/Isca stores spectral coefficients in rectangular arrays indexed
``(m, n)`` with ``m = 0 … num_fourier`` (zonal wavenumber) and
``n = 0 … num_spherical`` an *offset* index: the total (spherical)
wavenumber is ``l = m + n`` (see ``l2 = (m+n)**2`` in
``gauss_and_legendre.F90``). For triangular truncation T_M:

* ``num_fourier = M`` (42 for T42),
* ``num_spherical = M + 1``,
* prognostic fields live on the triangle ``l = m + n <= M``,
* the extra diagonal ``l = M + 1`` exists to support meridional-derivative
  recurrences (as in the Fortran core).

Arrays here are numpy (grid setup is host-side, float64, done once);
everything is a plain array so it drops into a JAX pytree unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Truncation:
    """Triangular truncation T_{num_fourier} with GFDL storage layout."""

    num_fourier: int  # M, e.g. 42
    fourier_inc: int = 1

    @property
    def num_spherical(self) -> int:
        return self.num_fourier + 1

    @property
    def shape(self) -> tuple[int, int]:
        """Storage shape (m, n) = (M+1, M+2)."""
        return (self.num_fourier + 1, self.num_spherical + 1)


def total_wavenumber(trunc: Truncation) -> np.ndarray:
    """l = m + n on the storage grid, shape ``trunc.shape``, int64."""
    m = np.arange(trunc.num_fourier + 1)[:, None]
    n = np.arange(trunc.num_spherical + 1)[None, :]
    return m + n


def prognostic_mask(trunc: Truncation) -> np.ndarray:
    """True where l <= M — the coefficients a prognostic field may occupy."""
    return total_wavenumber(trunc) <= trunc.num_fourier


def storage_mask(trunc: Truncation) -> np.ndarray:
    """True where l <= M + 1 — includes the derivative diagonal."""
    return total_wavenumber(trunc) <= trunc.num_spherical


def laplacian_eigenvalues(trunc: Truncation, radius: float) -> np.ndarray:
    """Eigenvalues of the horizontal Laplacian: -l(l+1)/a^2, shape ``trunc.shape``."""
    l = total_wavenumber(trunc).astype(np.float64)
    return -l * (l + 1.0) / (radius * radius)


@dataclass(frozen=True)
class SpectralGrid:
    """Static description of a (truncation, Gaussian grid, longitude) triple."""

    truncation: Truncation
    nlat: int
    nlon: int
    radius: float

    def __post_init__(self) -> None:
        # Unaliased quadrature for quadratic terms requires the standard
        # T_M <-> grid pairing; warn-level enforcement only (linear transforms
        # are exact whenever nlat >= (2M+1)/2 and nlon > 2M).
        m = self.truncation.num_fourier
        if self.nlon <= 2 * m:
            raise ValueError(f"nlon={self.nlon} cannot represent m up to {m}")
        if 2 * self.nlat < 2 * m + 1:
            raise ValueError(f"nlat={self.nlat} too small for exact analysis at T{m}")


T42_GRID = SpectralGrid(Truncation(42), nlat=64, nlon=128, radius=6376.0e3)
T85_GRID = SpectralGrid(Truncation(85), nlat=128, nlon=256, radius=6376.0e3)
