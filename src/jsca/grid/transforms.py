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
import numpy as np

from .gaussian import GaussianGrid, gaussian_grid
from .legendre import compute_legendre
from .spectral import (
    SpectralGrid,
    Truncation,
    laplacian_eigenvalues,
    prognostic_mask,
    storage_mask,
    total_wavenumber,
)


class SphericalCoeffs(NamedTuple):
    """Recurrence coefficients for the spectral vector operators.

    Port of the ``coef_*`` arrays built in ``spherical.F90`` ``spherical_init``
    (lines 183-214). All are real ``(M+1, N+1)`` with the GFDL storage
    convention ``spherical_wave l = m*fourier_inc + n``, ``fourier_wave = m*fourier_inc``:

    * ``dx``   ``= l_fourier / a``                         (lon derivative)
    * ``dym``  ``= (l-1) eps / a``, ``dyp[:, n] = (l+2) eps[:, n+1] / a``  (lat derivative)
    * ``uvm``  ``= -a eps / l``, ``uvc = -a l_fourier / (l(l+1))``,
      ``uvp[:, n] = -a eps[:, n+1] / (l+1)``               (vor,div -> u cos, v cos)
    * ``alpm`` ``= (l+1) eps / a``, ``alpp[:, n] = l eps[:, n+1] / a``     (alpha operator)

    with ``eps(m, n) = sqrt((l^2 - l_fourier^2)/(4 l^2 - 1))``. ``uvm``/``uvc`` are
    zero at ``l = 0``; the ``*p`` arrays are defined for ``n = 0 … N-1`` (their last
    column is unused by the operators and held at 0).
    """

    dx: jnp.ndarray
    dym: jnp.ndarray
    dyp: jnp.ndarray
    uvm: jnp.ndarray
    uvc: jnp.ndarray
    uvp: jnp.ndarray
    alpm: jnp.ndarray
    alpp: jnp.ndarray


def spherical_coeffs(trunc: Truncation, radius: float) -> SphericalCoeffs:
    """Build the ``coef_*`` recurrence tables (host-side, float64) — ``spherical_init``."""
    a = radius
    fw = (np.arange(trunc.num_fourier + 1)[:, None] * trunc.fourier_inc).astype(np.float64)
    sw = total_wavenumber(trunc).astype(np.float64)  # l = m*fourier_inc + n
    fw = np.broadcast_to(fw, sw.shape).copy()
    eps = np.sqrt((sw**2 - fw**2) / (4.0 * sw**2 - 1.0))  # 0 at l=0 (0/-1)

    sw_pos = sw > 0
    sw_safe = np.where(sw_pos, sw, 1.0)
    uvm = np.where(sw_pos, -a * eps / sw_safe, 0.0)
    uvc = np.where(sw_pos, -a * fw / (sw_safe * (sw_safe + 1.0)), 0.0)

    uvp = np.zeros_like(sw)
    alpp = np.zeros_like(sw)
    dyp = np.zeros_like(sw)
    uvp[:, :-1] = -a * eps[:, 1:] / (sw[:, :-1] + 1.0)
    alpp[:, :-1] = sw[:, :-1] * eps[:, 1:] / a
    dyp[:, :-1] = (sw[:, :-1] + 2.0) * eps[:, 1:] / a

    return SphericalCoeffs(
        dx=jnp.asarray(fw / a),
        dym=jnp.asarray((sw - 1.0) * eps / a),
        dyp=jnp.asarray(dyp),
        uvm=jnp.asarray(uvm),
        uvc=jnp.asarray(uvc),
        uvp=jnp.asarray(uvp),
        alpm=jnp.asarray((sw + 1.0) * eps / a),
        alpp=jnp.asarray(alpp),
    )


class TransformParams(NamedTuple):
    """Precomputed transform operators for one ``SpectralGrid`` (a JAX pytree)."""

    legendre: jnp.ndarray  # (nlat, M+1, N+1) float64
    legendre_wts: jnp.ndarray  # legendre * w_j[:, None, None], analysis operator
    sin_lat: jnp.ndarray  # (nlat,)
    wts_lat: jnp.ndarray  # (nlat,)
    mask_prognostic: jnp.ndarray  # (M+1, N+1) bool: l <= M
    mask_storage: jnp.ndarray  # (M+1, N+1) bool: l <= M+1
    lap_eig: jnp.ndarray  # (M+1, N+1) float64: -l(l+1)/a^2
    coeffs: SphericalCoeffs  # spectral vector-operator recurrence tables
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
        coeffs=spherical_coeffs(grid.truncation, grid.radius),
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


# --- spherical vector operators (port of spherical.F90) ----------------------
#
# Faithful port of the spectral vector operators in
# ``src/atmos_spectral/tools/spherical.F90``. Spectral fields are ``(..., m, n)``
# (GFDL storage, total wavenumber ``l = m + n``) with ``(m, n)`` the last two
# axes — the same layout as :func:`grid_to_spectral` and :func:`laplacian`, so
# leading axes (levels/batch) broadcast against the ``(M+1, N+1)`` coefficients.
# The meridional recurrences couple ``n`` with ``n±1`` and use the extra ``l=M+1``
# storage diagonal, exactly as the Fortran. ``i*z`` is the Fortran
# ``cmplx(-aimag(z), real(z))``.


def _shift_up(x: jnp.ndarray) -> jnp.ndarray:
    """``result[..., n] = x[..., n+1]``; 0 in the last slot (the ``[:, :-1] = [:, 1:]`` shift)."""
    return jnp.concatenate([x[..., 1:], jnp.zeros_like(x[..., :1])], axis=-1)


def _shift_down(x: jnp.ndarray) -> jnp.ndarray:
    """``result[..., n] = x[..., n-1]``; 0 in the first slot (the ``[:, 1:] = [:, :-1]`` shift)."""
    return jnp.concatenate([jnp.zeros_like(x[..., :1]), x[..., :-1]], axis=-1)


def compute_lon_deriv_cos(params: TransformParams, spec: jnp.ndarray) -> jnp.ndarray:
    """Cos-weighted longitude derivative — port of ``compute_lon_deriv_cos``."""
    return params.coeffs.dx * (1j * spec)


def compute_lat_deriv_cos(params: TransformParams, spec: jnp.ndarray) -> jnp.ndarray:
    """Cos-weighted latitude (∂/∂μ) derivative — port of ``compute_lat_deriv_cos``.

    ``deriv[..., n] = -dym[n] spec[n-1] + dyp[n] spec[n+1]`` (using the ``l=M+1``
    diagonal); the ``n=0`` slot has no lower neighbour and the ``n=M+1`` slot no upper.
    """
    c = params.coeffs
    return -c.dym * _shift_down(spec) + c.dyp * _shift_up(spec)


def compute_gradient_cos(
    params: TransformParams, spec: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(deriv_lon, deriv_lat)`` — port of ``compute_gradient_cos``."""
    return compute_lon_deriv_cos(params, spec), compute_lat_deriv_cos(params, spec)


def compute_laplacian(
    params: TransformParams, spec: jnp.ndarray, power: int | None = None
) -> jnp.ndarray:
    """Spectral Laplacian (optionally to an integer ``power``) — port of ``compute_laplacian``.

    ``factor = (-l(l+1)/a^2)^power``; ``power=None`` is the plain Laplacian.
    For ``power < 0`` the ``l=0`` coefficient (where the eigenvalue is 0) is set to 0.
    """
    lap = params.lap_eig  # -l(l+1)/a^2
    if power is None:
        factor = lap
    elif power >= 0:
        factor = lap**power
    else:
        factor = jnp.where(lap != 0.0, jnp.where(lap != 0.0, lap, 1.0) ** power, 0.0)
    return spec * factor


def compute_ucos_vcos(
    params: TransformParams, vorticity: jnp.ndarray, divergence: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(vor, div) -> (u cos, v cos)`` — port of ``compute_ucos_vcos``."""
    c = params.coeffs
    u_cos = (
        c.uvc * (1j * divergence)
        + c.uvm * _shift_down(vorticity)
        - c.uvp * _shift_up(vorticity)
    )
    v_cos = (
        c.uvc * (1j * vorticity)
        - c.uvm * _shift_down(divergence)
        + c.uvp * _shift_up(divergence)
    )
    return u_cos, v_cos


def _alpha_operator(
    params: TransformParams, spec_a: jnp.ndarray, spec_b: jnp.ndarray, isign: int
) -> jnp.ndarray:
    """Port of the private ``compute_alpha_operator`` (spherical.F90 L513-561)."""
    c = params.coeffs
    return (
        c.dx * (1j * spec_a)
        - isign * c.alpm * _shift_down(spec_b)
        + isign * c.alpp * _shift_up(spec_b)
    )


def compute_vor(
    params: TransformParams, u_cos: jnp.ndarray, v_cos: jnp.ndarray
) -> jnp.ndarray:
    """Vorticity from ``(u cos, v cos)`` — port of ``compute_vor`` (``alpha(v, u, -1)``)."""
    return _alpha_operator(params, v_cos, u_cos, -1)


def compute_div(
    params: TransformParams, u_cos: jnp.ndarray, v_cos: jnp.ndarray
) -> jnp.ndarray:
    """Divergence from ``(u cos, v cos)`` — port of ``compute_div`` (``alpha(u, v, +1)``)."""
    return _alpha_operator(params, u_cos, v_cos, +1)


def compute_vor_div(
    params: TransformParams, u_cos: jnp.ndarray, v_cos: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(u cos, v cos) -> (vorticity, divergence)`` — port of ``compute_vor_div``."""
    return compute_vor(params, u_cos, v_cos), compute_div(params, u_cos, v_cos)
