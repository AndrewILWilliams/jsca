import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.grid import (
    T42_GRID,
    build_transforms,
    grid_to_spectral,
    hyperdiffusion_multiplier,
    laplacian,
    spectral_to_grid,
)

PARAMS, GG = build_transforms(T42_GRID)


def random_prognostic_spec(rng, nlev=None):
    """Random spectral field on the prognostic triangle (l <= M), m=0 real."""
    shape = (43, 44) if nlev is None else (nlev, 43, 44)
    spec = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    spec[..., 0, :] = spec[..., 0, :].real  # m = 0: real for a real field
    return jnp.asarray(spec * np.asarray(PARAMS.mask_prognostic))


def test_x64_active():
    assert jax.config.jax_enable_x64
    assert PARAMS.legendre.dtype == jnp.float64


def test_spectral_grid_spectral_roundtrip():
    rng = np.random.default_rng(0)
    spec = random_prognostic_spec(rng)
    grid = spectral_to_grid(PARAMS, spec)
    assert grid.dtype == jnp.float64
    spec2 = grid_to_spectral(PARAMS, grid) * PARAMS.mask_prognostic
    np.testing.assert_allclose(np.asarray(spec2), np.asarray(spec), atol=1e-12)


def test_grid_spectral_grid_roundtrip_band_limited():
    rng = np.random.default_rng(1)
    grid = spectral_to_grid(PARAMS, random_prognostic_spec(rng))  # band-limited by construction
    grid2 = spectral_to_grid(PARAMS, grid_to_spectral(PARAMS, grid))
    np.testing.assert_allclose(np.asarray(grid2), np.asarray(grid), atol=1e-12)


def test_batched_levels_roundtrip():
    rng = np.random.default_rng(2)
    spec = random_prognostic_spec(rng, nlev=25)
    grid = spectral_to_grid(PARAMS, spec)
    assert grid.shape == (25, 64, 128)
    spec2 = grid_to_spectral(PARAMS, grid) * PARAMS.mask_prognostic
    np.testing.assert_allclose(np.asarray(spec2), np.asarray(spec), atol=1e-12)


def test_global_mean_invariant():
    """Area mean of the grid field equals spec[0,0] * P~00 = spec[0,0]/sqrt(2)."""
    rng = np.random.default_rng(3)
    spec = random_prognostic_spec(rng)
    grid = np.asarray(spectral_to_grid(PARAMS, spec))
    area_mean = np.einsum("j,ji->", np.asarray(PARAMS.wts_lat), grid) / (2.0 * grid.shape[-1])
    np.testing.assert_allclose(area_mean, float(spec[0, 0].real) * np.sqrt(0.5), atol=1e-13)


def test_laplacian_eigenfunction():
    """Y_l^m is an eigenfunction: lap = -l(l+1)/a^2."""
    m, n = 5, 7  # l = 12
    l = m + n
    spec = jnp.zeros((43, 44), dtype=jnp.complex128).at[m, n].set(1.0 + 0.5j)
    lap = laplacian(PARAMS, spec)
    expected = -l * (l + 1) / T42_GRID.radius**2
    np.testing.assert_allclose(complex(lap[m, n]), (1.0 + 0.5j) * expected, rtol=1e-15)
    assert np.count_nonzero(np.asarray(lap)) == 1


def test_hyperdiffusion_multiplier_del8():
    mult = np.asarray(hyperdiffusion_multiplier(PARAMS, order=8, coefficient=1.0e-2))
    assert mult.shape == (43, 44)
    assert mult[0, 0] == 0.0
    assert np.all(mult <= 0)
    # scales as [l(l+1)]^4
    l1, l2 = 10, 20
    ratio = mult[0, l2] / mult[0, l1]
    np.testing.assert_allclose(ratio, ((l2 * (l2 + 1)) / (l1 * (l1 + 1))) ** 4, rtol=1e-12)


def test_jit_and_vmap_compose():
    rng = np.random.default_rng(4)
    batch = jnp.stack([spectral_to_grid(PARAMS, random_prognostic_spec(rng)) for _ in range(3)])
    fwd = jax.jit(lambda f: grid_to_spectral(PARAMS, f))
    single = fwd(batch[0])
    stacked = jax.vmap(fwd)(batch)
    np.testing.assert_allclose(np.asarray(stacked[0]), np.asarray(single), atol=0)


@pytest.mark.parametrize("m,n", [(0, 3), (4, 0), (10, 11)])
def test_pure_harmonic_synthesis(m, n):
    """Synthesis of a single coefficient gives P~(mu) * cos/sin structure in lon."""
    spec = jnp.zeros((43, 44), dtype=jnp.complex128).at[m, n].set(1.0)
    grid = np.asarray(spectral_to_grid(PARAMS, spec))
    lam = 2 * np.pi * np.arange(128) / 128
    pnm = np.asarray(PARAMS.legendre)[:, m, n]
    factor = 1.0 if m == 0 else 2.0  # m>0 appears with its conjugate pair
    expected = factor * np.outer(pnm, np.cos(m * lam))
    np.testing.assert_allclose(grid, expected, atol=1e-13)
