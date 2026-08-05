"""Validation for the wind <-> vorticity/divergence grid transforms and
horizontal advection.

``uv_grid_from_vor_div`` / ``vor_div_from_uv_grid`` / ``horizontal_advection``
(transforms.F90 L700/L742/L808) are thin compositions of the spectral vector
operators ``compute_ucos_vcos`` / ``compute_vor_div`` / ``compute_gradient_cos``
— each already Tier-1 fixture-validated against real Fortran
(``spherical_reference``) — with ``spectral_to_grid`` / ``grid_to_spectral`` and
``divide_by_cos``. Rather than stand up the full Fortran transform stack for a new
golden, they are validated **analytically and by round-trip**:

* solid-body rotation ``u = U cos(lat), v = 0`` has the exact vorticity
  ``2 U sin(lat) / a`` and zero divergence;
* ``u, v -> vor, div -> u, v`` recovers a band-limited wind to machine precision
  (with a de-aliasing buffer, as in any spectral model);
* advecting a known field by a known wind matches the exact ``-u.grad(field)``.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jsca  # noqa: F401
from jsca import constants
from jsca.grid.spectral import SpectralGrid, Truncation, total_wavenumber
from jsca.grid.transforms import (
    build_transforms,
    grid_to_spectral,
    horizontal_advection,
    spectral_to_grid,
    uv_grid_from_vor_div,
    vor_div_from_uv_grid,
)

NF = 42
RADIUS = constants.RADIUS
GRID = SpectralGrid(Truncation(NF), nlat=2 * NF + 2, nlon=4 * NF + 4, radius=RADIUS)
PARAMS, _GG = build_transforms(GRID)


def _coords():
    mu = np.asarray(PARAMS.sin_lat)
    lon = 2.0 * np.pi * np.arange(GRID.nlon) / GRID.nlon
    lon2d, mu2d = np.meshgrid(lon, mu)
    return lon2d, mu2d, np.sqrt(1.0 - mu2d**2)


def test_solid_body_rotation_vorticity():
    """u = U cos(lat), v = 0  ->  vor = 2 U sin(lat)/a, div = 0."""
    _lon, mu, cos = _coords()
    u = 30.0 * cos
    v = np.zeros_like(u)
    vor, div = vor_div_from_uv_grid(PARAMS, jnp.asarray(u), jnp.asarray(v))
    vor_g = np.asarray(spectral_to_grid(PARAMS, vor))
    div_g = np.asarray(spectral_to_grid(PARAMS, div))
    np.testing.assert_allclose(vor_g, 2.0 * 30.0 * mu / RADIUS, rtol=1e-11, atol=1e-16)
    assert np.max(np.abs(div_g)) < 1e-16


def test_wind_roundtrip():
    """u, v -> vor, div -> u, v recovers a band-limited wind to machine precision."""
    lvals = np.asarray(total_wavenumber(Truncation(NF)))
    rng = np.random.default_rng(2)
    z = (rng.standard_normal(lvals.shape) + 1j * rng.standard_normal(lvals.shape)) * 1e-6
    z[(lvals > 15) | (lvals == 0)] = 0.0  # low-wavenumber (de-aliased) field
    vor0, div0 = jnp.asarray(z), jnp.asarray(z * 0.5)
    u, v = uv_grid_from_vor_div(PARAMS, vor0, div0)
    vor, div = vor_div_from_uv_grid(PARAMS, u, v)
    u2, v2 = uv_grid_from_vor_div(PARAMS, vor, div)
    np.testing.assert_allclose(np.asarray(u2), np.asarray(u), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(v2), np.asarray(v), rtol=1e-10, atol=1e-12)


def test_horizontal_advection_analytic():
    """-u.grad of field = cos(lat)cos(lon) advected by u = U cos(lat), v = 0."""
    lon, _mu, cos = _coords()
    u = 30.0 * cos
    v = np.zeros_like(u)
    field = cos * np.cos(lon)
    field_spec = grid_to_spectral(PARAMS, jnp.asarray(field))
    tend = horizontal_advection(
        PARAMS, field_spec, jnp.asarray(u), jnp.asarray(v), jnp.zeros_like(jnp.asarray(field))
    )
    # -u * (1/(a cos)) dfield/dlon ; dfield/dlon = -cos(lat) sin(lon)
    analytic = -(u) * (-cos * np.sin(lon)) / (RADIUS * cos)
    np.testing.assert_allclose(np.asarray(tend), analytic, rtol=1e-11, atol=1e-14)


def test_rhomboidal_not_supported():
    import pytest

    _lon, _mu, cos = _coords()
    u = jnp.asarray(cos)
    with pytest.raises(NotImplementedError):
        vor_div_from_uv_grid(PARAMS, u, jnp.zeros_like(u), triang=False)


def test_jit():
    _lon, _mu, cos = _coords()
    u = jnp.asarray(30.0 * cos)
    v = jnp.zeros_like(u)
    eager = vor_div_from_uv_grid(PARAMS, u, v)
    jitted = jax.jit(lambda a, b: vor_div_from_uv_grid(PARAMS, a, b))(u, v)
    for e, j in zip(eager, jitted):
        # near-zero coefficients differ by fp noise (~1e-21) between eager/jit
        np.testing.assert_allclose(np.asarray(e), np.asarray(j), rtol=1e-11, atol=1e-16)
