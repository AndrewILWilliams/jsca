"""Validation for compute_pressure_gradient.

``compute_pressure_gradient`` (spectral_dynamics.F90 L1192-1209) is a thin
composition of ``compute_gradient_cos`` — which is itself Tier-1 fixture-validated
against the real Fortran (``tests/test_spherical_fixtures.py`` /
``spherical_reference.npz``) — and ``spectral_to_grid``, plus an elementwise
``* psg`` and ``/ cos(lat)``. Rather than stand up the full Fortran transform
stack for a new golden, it is validated **analytically**: the spectral gradient
of a band-limited ``ln(ps)`` is exact, so ``grad(psg) = psg grad(ln ps)`` has a
closed form on the Gaussian grid. The test confirms both components equal that
exact analytic gradient (the zonal ``(1/(a cos)) d/dlon`` and meridional
``(1/a) d/dtheta`` of ``psg``) to spectral accuracy.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jsca  # noqa: F401
from jsca import constants
from jsca.grid.spectral import SpectralGrid, Truncation
from jsca.grid.transforms import (
    build_transforms,
    compute_pressure_gradient,
    grid_to_spectral,
)

NF = 21
RADIUS = constants.RADIUS
GRID = SpectralGrid(Truncation(NF), nlat=2 * NF + 2, nlon=4 * NF + 4, radius=RADIUS)
PARAMS, _GG = build_transforms(GRID)


def _grid_coords():
    mu = np.asarray(PARAMS.sin_lat)  # sin(lat) at the Gaussian latitudes
    lon = 2.0 * np.pi * np.arange(GRID.nlon) / GRID.nlon
    lon2d, mu2d = np.meshgrid(lon, mu)  # (nlat, nlon)
    return lon2d, mu2d, np.sqrt(1.0 - mu2d**2)  # lon, mu, cos(lat)


def test_zonal_symmetric_has_zero_zonal_gradient():
    """A zonally symmetric ln(ps) (= alpha*sin(lat)) has an identically zero zonal
    pressure gradient (only m=0 coefficients contribute)."""
    _lon, mu, _cos = _grid_coords()
    lnps = 0.02 * mu
    lnps_spec = grid_to_spectral(PARAMS, jnp.asarray(lnps))
    dx, _dy = compute_pressure_gradient(PARAMS, lnps_spec, jnp.asarray(np.exp(lnps)))
    assert np.max(np.abs(np.asarray(dx))) < 1e-12


def test_matches_analytic_gradient():
    """Both components equal the exact analytic grad(psg) for a band-limited (l=1)
    ln(ps) = alpha*sin(lat) + beta*cos(lat)*cos(lon)."""
    lon, mu, cos = _grid_coords()
    alpha, beta = 0.02, 0.015
    lnps = alpha * mu + beta * cos * np.cos(lon)
    psg = np.exp(lnps)
    lnps_spec = grid_to_spectral(PARAMS, jnp.asarray(lnps))
    dx, dy = compute_pressure_gradient(PARAMS, lnps_spec, jnp.asarray(psg))

    dlnps_dlon = -beta * cos * np.sin(lon)
    dlnps_dtheta = alpha * cos - beta * mu * np.cos(lon)  # d/dtheta, mu = sin(theta)
    dx_analytic = psg * dlnps_dlon / (RADIUS * cos)  # (1/(a cos)) d psg / d lon
    dy_analytic = psg * dlnps_dtheta / RADIUS  # (1/a) d psg / d theta

    np.testing.assert_allclose(np.asarray(dx), dx_analytic, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(np.asarray(dy), dy_analytic, rtol=1e-11, atol=1e-12)


def test_jit():
    lon, mu, cos = _grid_coords()
    lnps = 0.02 * mu + 0.015 * cos * np.cos(lon)
    lnps_spec = grid_to_spectral(PARAMS, jnp.asarray(lnps))
    psg = jnp.asarray(np.exp(lnps))
    eager = compute_pressure_gradient(PARAMS, lnps_spec, psg)
    # params is a closed-over constant (its integer nlon must stay concrete for
    # the FFT), exactly as the model uses it.
    jitted = jax.jit(lambda spec, p: compute_pressure_gradient(PARAMS, spec, p))(lnps_spec, psg)
    for e, j in zip(eager, jitted):
        np.testing.assert_allclose(np.asarray(e), np.asarray(j), rtol=1e-14, atol=0.0)
