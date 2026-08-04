"""Spherical-harmonic transforms: FFT in longitude, Legendre matmul in latitude.

This is the JAX-side engine corresponding to Isca's
``src/atmos_spectral/tools/{grid_fourier, spherical_fourier, transforms}.F90``.

Conventions (documented so Fortran fixtures can be matched later):

* Physical expansion (matches the scoping doc):
      X(lambda_i, mu_j) = sum_{m=-M}^{M} sum_l X_l^m P~_l^m(mu_j) e^{i m lambda_i}
  with P~ the unit-integral ALFs of ``jsca.grid.legendre`` and, for real
  fields, X_l^{-m} = conj(X_l^m); only m >= 0 is stored.
* Longitude analysis: F_m(mu_j) = (1/nlon) * sum_i X e^{-i m lambda_i}
  (``rfft / nlon``). Synthesis is the exact inverse (``irfft`` of
  ``F * nlon``). GFDL's fft99 applies its 1/n on the same (analysis) side.
* Latitude analysis is Gaussian quadrature:
      X_l^m = sum_j w_j F_m(mu_j) P~_l^m(mu_j)
  — no extra factor, because sum_j w_j = 2 and integral of P~^2 dmu = 1.

Precision: transform matrices are built in float64 (numpy) once, then used
inside jitted JAX code. Run with ``jax.config.update("jax_enable_x64", True)``
— enforced at import of :mod:`jsca` for now (Phase-0 policy: validate in f64).

All functions are pure; ``TransformParams`` is a NamedTuple-of-arrays pytree,
so the whole module composes with ``jax.jit``/``vmap``/``scan``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .gaussian import GaussianGrid, gaussian_grid
from .legendre import compute_legendre
from .spectral import SpectralGrid, laplacian_eigenvalues, prognostic_mask, storage_mask


class TransformParams(NamedTuple):
    """Precomputed transform operators for one ``SpectralGrid`` (a JAX pytree)."""

    legendre: jnp.ndarray  # (nlat, M+1, N+1) float64
    legendre_wts: jnp.ndarray  # legendre * w_j[:, None, None], analysis operator
    sin_lat: jnp.ndarray  # (nlat,)
    wts_lat: jnp.ndarray  # (nlat,)
    mask_prognostic: jnp.ndarray  # (M+1, N+1) bool: l <= M
    mask_storage: jnp.ndarray  # (M+1, N+1) bool: l <= M+1
    lap_eig: jnp.ndarray  # (M+1, N+1) float64: -l(l+1)/a^2
    nlon: int
    num_fourier: int


def build_transforms(grid: SpectralGrid) -> tuple[TransformParams, GaussianGrid]:
    """Build all static operators for ``grid`` (host-side, float64, once)."""
    gg = gaussian_grid(grid.nlat)
    leg = compute_legendre(grid.truncation, gg.sin_lat)
    params = TransformParams(
        legendre=jnp.asarray(leg),
        legendre_wts=jnp.asarray(leg * gg.wts_lat[:, None, None]),
        sin_lat=jnp.asarray(gg.sin_lat),
        wts_lat=jnp.asarray(gg.wts_lat),
        mask_prognostic=jnp.asarray(prognostic_mask(grid.truncation)),
        mask_storage=jnp.asarray(storage_mask(grid.truncation)),
        lap_eig=jnp.asarray(laplacian_eigenvalues(grid.truncation, grid.radius)),
        nlon=grid.nlon,
        num_fourier=grid.truncation.num_fourier,
    )
    return params, gg


# --- longitude (Fourier) leg -------------------------------------------------


def grid_to_fourier(params: TransformParams, field: jnp.ndarray) -> jnp.ndarray:
    """(..., nlat, nlon) real -> (..., nlat, M+1) complex; F_m = rfft/nlon."""
    fm = jnp.fft.rfft(field, axis=-1) / params.nlon
    return fm[..., : params.num_fourier + 1]

def fourier_to_grid(params: TransformParams, fm: jnp.ndarray) -> jnp.ndarray:
    """(..., nlat, M+1) complex -> (..., nlat, nlon) real. Exact inverse of
    :func:`grid_to_fourier` for band-limited fields (m > M zero-padded)."""
    nfreq = params.nlon // 2 + 1
    pad = [(0, 0)] * (fm.ndim - 1) + [(0, nfreq - fm.shape[-1])]
    fm_full = jnp.pad(fm, pad)
    return jnp.fft.irfft(fm_full * params.nlon, n=params.nlon, axis=-1)


# --- latitude (Legendre) leg -------------------------------------------------


def fourier_to_spectral(params: TransformParams, fm: jnp.ndarray) -> jnp.ndarray:
    """(..., nlat, M+1) -> (..., M+1, N+1): X_l^m = sum_j w_j F_m(mu_j) P~(mu_j)."""
    spec = jnp.einsum("...jm,jmn->...mn", fm, params.legendre_wts)
    return jnp.where(params.mask_storage, spec, 0.0)

def spectral_to_fourier(params: TransformParams, spec: jnp.ndarray) -> jnp.ndarray:
    """(..., M+1, N+1) -> (..., nlat, M+1): F_m(mu_j) = sum_n X P~."""
    return jnp.einsum("...mn,jmn->...jm", spec, params.legendre)


# --- full transforms ---------------------------------------------------------


def grid_to_spectral(params: TransformParams, field: jnp.ndarray) -> jnp.ndarray:
    """Analysis: (..., nlat, nlon) real -> (..., M+1, N+1) complex spectral."""
    return fourier_to_spectral(params, grid_to_fourier(params, field))

def spectral_to_grid(params: TransformParams, spec: jnp.ndarray) -> jnp.ndarray:
    """Synthesis: (..., M+1, N+1) complex -> (..., nlat, nlon) real grid."""
    return fourier_to_grid(params, spectral_to_fourier(params, spec))


# --- spectral-space operators ------------------------------------------------


def laplacian(params: TransformParams, spec: jnp.ndarray) -> jnp.ndarray:
    """Horizontal Laplacian in spectral space: multiply by -l(l+1)/a^2."""
    return spec * params.lap_eig

def hyperdiffusion_multiplier(
    params: TransformParams, order: int, coefficient: float
) -> jnp.ndarray:
    """Damping multiplier -nu * [l(l+1)/a^2]^(order/2) per coefficient.

    ``order`` is the del-power (Isca's ``2 * damping_order``; default del^8
    -> order=8). Returned as a real (M+1, N+1) array to multiply tendencies.
    """
    return -coefficient * (-params.lap_eig) ** (order // 2)
