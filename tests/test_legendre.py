import numpy as np
import pytest

from jsca.grid import Truncation, compute_legendre, epsilon, gaussian_grid, total_wavenumber

T42 = Truncation(42)


def test_epsilon_known_values():
    eps = epsilon(T42)
    # eps(m=0, n=1): l=1 -> sqrt((1-0)/(4-1)) = 1/sqrt(3)
    np.testing.assert_allclose(eps[0, 1], 1.0 / np.sqrt(3.0), rtol=1e-15)
    # eps(m=2, n=1): l=3 -> sqrt((9-4)/(36-1)) = sqrt(5/35)
    np.testing.assert_allclose(eps[2, 1], np.sqrt(5.0 / 35.0), rtol=1e-15)


def test_low_order_explicit_values():
    """P~ with unit integral: P~00=sqrt(1/2), P~10=sqrt(3/2)mu,
    P~11=sqrt(3)/2*cos(theta), P~20=sqrt(5/2)(3mu^2-1)/2 (no Condon-Shortley)."""
    mu = np.array([-0.7, -0.2, 0.0, 0.3, 0.9])
    p = compute_legendre(T42, mu)  # (nlat, m, n); l = m + n
    cos = np.sqrt(1 - mu**2)
    np.testing.assert_allclose(p[:, 0, 0], np.sqrt(0.5), rtol=1e-15)
    np.testing.assert_allclose(p[:, 0, 1], np.sqrt(1.5) * mu, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(p[:, 1, 0], np.sqrt(0.75) * cos, rtol=1e-14)
    np.testing.assert_allclose(
        p[:, 0, 2], np.sqrt(2.5) * (3 * mu**2 - 1) / 2, rtol=1e-14, atol=1e-15
    )
    # no Condon-Shortley phase: sectoral values are positive
    assert np.all(p[2, np.arange(0, 43), 0] > 0)


@pytest.mark.parametrize("m", [0, 1, 5, 21, 42])
def test_orthonormality_on_gaussian_grid(m):
    """sum_j w_j P~(l,m) P~(l',m) = delta_{ll'} — exact for l+l' <= 2*nlat-1."""
    gg = gaussian_grid(64)
    p = compute_legendre(T42, gg.sin_lat)  # (64, 43, 44)
    valid_n = 43 - m + 1  # l = m+n <= 43 (storage triangle incl. derivative row)
    block = p[:, m, :valid_n]  # (nlat, n)
    gram = np.einsum("j,jn,jk->nk", gg.wts_lat, block, block)
    np.testing.assert_allclose(gram, np.eye(valid_n), atol=2e-13)


def test_orthonormality_independent_quadrature():
    """Same, but on numpy's 256-point rule — independent of jsca's grid code."""
    nodes, weights = np.polynomial.legendre.leggauss(256)
    p = compute_legendre(T42, nodes)
    for m in (0, 7, 42):
        valid_n = 43 - m + 1
        block = p[:, m, :valid_n]
        gram = np.einsum("j,jn,jk->nk", weights, block, block)
        np.testing.assert_allclose(gram, np.eye(valid_n), atol=2e-13)


def test_storage_triangle():
    lw = total_wavenumber(T42)
    assert lw.shape == (43, 44)
    assert lw[42, 1] == 43  # derivative row reachable at every m
