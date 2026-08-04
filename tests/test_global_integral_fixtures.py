"""Tier-1 tests for global_integral against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_global_integral_reference.F90``,
which compiles Isca's actual ``global_integral.F90`` unmodified, linked against
the real ``press_and_geopot.F90`` and ``gauss_and_legendre.F90`` (transforms_mod
stubbed: ``area_weighted_global_mean`` replicates transforms.F90's formula
verbatim using the real Gaussian weights). Regeneration recipe in that file's
header.

Fortran grid storage is ``(lon, lat, level)``; the port keeps ``(lat, lon, level)``,
so fixtures transpose lon<->lat. The result is a global sum, so the port's
pairwise summation differs from the Fortran sequential sum by a few ULP — rtol
1e-13.
"""

from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.dycore import mass_weighted_global_integral
from jsca.grid import SpectralGrid, Truncation, area_weighted_global_mean, build_transforms

FIXTURE = Path(__file__).parent / "fixtures" / "global_integral_reference.npz"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="global_integral fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def params(fx):
    nlon, nlat = int(fx["gi_meta"][0]), int(fx["gi_meta"][1])
    # a T(nlat/2 - 1)-ish grid gives the right nlat/nlon; only wts_lat and nlon
    # are used by the area mean. Match the driver's Gaussian grid (nlat=8, nlon=16).
    m = nlat - 1  # ensures 2*nlat >= 2m+1 and nlon > 2m for the driver's 8x16
    grid = SpectralGrid(Truncation(min(m, nlon // 2 - 1)), nlat=nlat, nlon=nlon, radius=6376.0e3)
    p, _ = build_transforms(grid)
    return p


def latlon3(a):
    return np.moveaxis(a, 0, 1)  # (lon, lat, level) -> (lat, lon, level)


def test_wts_lat_matches_driver(fx, params):
    """The port's Gaussian weights equal the driver's compute_gaussian weights."""
    np.testing.assert_allclose(np.asarray(params.wts_lat), fx["gi_wts_lat"], rtol=1e-14)


def test_mass_weighted_global_integral(fx, params):
    field = latlon3(fx["gi_field"])
    ps = fx["gi_ps"].T  # (lon, lat) -> (lat, lon)
    out = mass_weighted_global_integral(params, fx["gi_pk"], fx["gi_bk"], field, ps)
    np.testing.assert_allclose(float(out), float(fx["gi_integral"][0]), rtol=1e-13)


def test_area_weighted_global_mean_of_constant(params):
    """The area mean of a constant field is that constant (weights are normalised)."""
    ones = np.ones((params.wts_lat.shape[0], params.nlon))
    mean = float(area_weighted_global_mean(params, 3.5 * ones))
    np.testing.assert_allclose(mean, 3.5, rtol=1e-13)
