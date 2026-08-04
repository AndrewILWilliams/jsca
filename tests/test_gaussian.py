import numpy as np
import pytest

from jsca.grid import gaussian_grid, gaussian_hemisphere


@pytest.mark.parametrize("nlat", [32, 64, 128])
def test_matches_numpy_leggauss(nlat):
    """Independent cross-check: faithful Newton port vs numpy's Gauss-Legendre."""
    nodes, weights = np.polynomial.legendre.leggauss(nlat)  # ascending
    gg = gaussian_grid(nlat)
    # Two different last-ulp paths to the same rule (Newton recurrence here,
    # numpy's own solver there): compare in *absolute* terms — that's what
    # quadrature accuracy depends on — at a few-ulp tolerance. The exactness
    # test below independently proves the rule integrates to 1e-13.
    np.testing.assert_allclose(gg.sin_lat, nodes, rtol=0, atol=5e-14)
    np.testing.assert_allclose(gg.wts_lat, weights, rtol=0, atol=5e-14)


def test_hemisphere_layout_matches_fortran():
    """compute_gaussian returns positive nodes, pole-first (descending)."""
    sin_hem, wts_hem = gaussian_hemisphere(32)
    assert np.all(sin_hem > 0)
    assert np.all(np.diff(sin_hem) < 0)
    assert np.all(wts_hem > 0)


def test_weights_sum_to_two():
    gg = gaussian_grid(64)
    np.testing.assert_allclose(gg.wts_lat.sum(), 2.0, rtol=1e-15)


@pytest.mark.parametrize("k", [0, 2, 10, 40, 126])
def test_quadrature_exactness(k):
    """Gaussian quadrature with nlat nodes integrates mu^k exactly for k <= 2*nlat-1."""
    gg = gaussian_grid(64)
    exact = 2.0 / (k + 1) if k % 2 == 0 else 0.0
    approx = np.sum(gg.wts_lat * gg.sin_lat**k)
    np.testing.assert_allclose(approx, exact, rtol=0, atol=1e-13)


def test_global_grid_south_to_north_symmetric():
    gg = gaussian_grid(64)
    assert np.all(np.diff(gg.sin_lat) > 0)
    np.testing.assert_allclose(gg.sin_lat, -gg.sin_lat[::-1], atol=0)
    np.testing.assert_allclose(gg.wts_lat, gg.wts_lat[::-1], atol=0)
    np.testing.assert_allclose(np.sin(gg.lat), gg.sin_lat, atol=1e-15)


def test_odd_nlat_rejected():
    with pytest.raises(ValueError):
        gaussian_grid(63)
