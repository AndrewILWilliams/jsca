"""Tier-1 tests for the spherical spectral operators against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_spherical_reference.F90``, which
compiles Isca's actual ``spherical.F90`` unmodified (spec_mpp_mod stubbed for
``get_spec_domain``; ``radius`` passed to ``spherical_init``). Regeneration recipe
in that file's header.

Fortran spectral storage is ``(m, n, k)``; the port keeps ``(m, n)`` last
(``transforms.py`` convention), so fixtures are transposed to ``(k, m, n)`` at the
comparison boundary. Operators are pure multiply/add of the precomputed
recurrence coefficients, so agreement is near-bitwise (rtol 1e-13).
"""

from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.grid import (
    SpectralGrid,
    Truncation,
    build_transforms,
    compute_div,
    compute_gradient_cos,
    compute_laplacian,
    compute_lat_deriv_cos,
    compute_lon_deriv_cos,
    compute_ucos_vcos,
    compute_vor,
    compute_vor_div,
    laplacian,
)

FIXTURE = Path(__file__).parent / "fixtures" / "spherical_reference.npz"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="spherical fixtures not generated")

RTOL = 1e-13
ATOL = 1e-15


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def params(fx):
    nf = int(fx["sph_meta"][0])
    radius = float(fx["sph_meta"][4])
    # nlat/nlon only feed the (unused-here) Legendre operators; pick a valid pair.
    grid = SpectralGrid(Truncation(nf), nlat=2 * nf + 2, nlon=4 * nf + 4, radius=radius)
    p, _ = build_transforms(grid)
    return p


def cplx(fx, stem):
    """Fortran (m, n, k) complex fixture -> port layout (k, m, n)."""
    z = fx[f"{stem}_re"] + 1j * fx[f"{stem}_im"]
    return np.moveaxis(z, -1, 0)


def close(a, b):
    np.testing.assert_allclose(np.asarray(a), b, rtol=RTOL, atol=ATOL)


def test_lon_deriv(fx, params):
    close(compute_lon_deriv_cos(params, cplx(fx, "sph_spec")), cplx(fx, "sph_dlon"))


def test_lat_deriv(fx, params):
    close(compute_lat_deriv_cos(params, cplx(fx, "sph_spec")), cplx(fx, "sph_dlat"))


def test_gradient(fx, params):
    dlon, dlat = compute_gradient_cos(params, cplx(fx, "sph_spec"))
    close(dlon, cplx(fx, "sph_grad_lon"))
    close(dlat, cplx(fx, "sph_grad_lat"))


def test_laplacian_default(fx, params):
    close(compute_laplacian(params, cplx(fx, "sph_spec")), cplx(fx, "sph_lap1"))


def test_laplacian_power2(fx, params):
    close(compute_laplacian(params, cplx(fx, "sph_spec"), power=2), cplx(fx, "sph_lap2"))


def test_laplacian_matches_existing_helper(fx, params):
    """compute_laplacian (default) is the same as the pre-existing ``laplacian``."""
    spec = cplx(fx, "sph_spec")
    close(compute_laplacian(params, spec), np.asarray(laplacian(params, spec)))


def test_ucos_vcos(fx, params):
    u, v = compute_ucos_vcos(params, cplx(fx, "sph_vor"), cplx(fx, "sph_div"))
    close(u, cplx(fx, "sph_ucos_out"))
    close(v, cplx(fx, "sph_vcos_out"))


def test_vor_div(fx, params):
    vor, div = compute_vor_div(params, cplx(fx, "sph_ucos2"), cplx(fx, "sph_vcos2"))
    close(vor, cplx(fx, "sph_vor_div_vor"))
    close(div, cplx(fx, "sph_vor_div_div"))


def test_vor(fx, params):
    u, v = cplx(fx, "sph_ucos2"), cplx(fx, "sph_vcos2")
    close(compute_vor(params, u, v), cplx(fx, "sph_vor_out"))


def test_div(fx, params):
    u, v = cplx(fx, "sph_ucos2"), cplx(fx, "sph_vcos2")
    close(compute_div(params, u, v), cplx(fx, "sph_div_out"))


def test_eigen_provenance(fx, params):
    """lap_eig is -l(l+1)/a^2; the default Laplacian factor matches the Fortran -eigen."""
    from jsca.grid import laplacian_eigenvalues

    nf = int(fx["sph_meta"][0])
    radius = float(fx["sph_meta"][4])
    np.testing.assert_allclose(
        np.asarray(params.lap_eig), laplacian_eigenvalues(Truncation(nf), radius), rtol=1e-14
    )
