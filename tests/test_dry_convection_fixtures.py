"""Tier-1 tests for the dry convection scheme against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_dry_convection_reference.F90``,
which compiles the **unmodified** ``dry_convection.f90`` (Schneider-Walker dry
adjustment; ``tau=21600``, ``gamma=1``) over a column set whose static stability
sweeps from sub-adiabatic (stable, no convection) through super-adiabatic
(surface-based deep adjustment), plus an elevated-inversion column that raises the
LCL off the ground. So CAPE/CIN, the LCL/LZB search (both surface-based and
elevated) and the energy-conserving adjustment are all exercised.

Pure arithmetic (a power-law parcel lift + logs), so jsca matches Isca to machine
precision: ``dt_tg`` to ~1e-18, CAPE to the log tolerance, and the ``lzb``/``lcl``
indices exactly (Fortran 1-based -> 0-based here).
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics import DryConvectionParams, dry_convection

FIXTURE = Path(__file__).parent / "fixtures" / "dry_convection_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="dry_convection fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def result(fx):
    # fixture namelist: tau=21600, gamma=1.0
    return [np.asarray(a) for a in dry_convection(
        DryConvectionParams(tau=21600.0, gamma=1.0),
        fx["dc_t"], fx["dc_pfull"], fx["dc_phalf"])]


def test_fixture_exercises_all_branches(fx):
    """Guard: the fixture has non-convecting, surface-based, and elevated-LCL cols."""
    lcl = fx["dc_lcl"].astype(int)
    lzb = fx["dc_lzb"].astype(int)
    K = fx["dc_t"].shape[-1]
    assert (lzb == K).any()                 # some columns never convect (lzb=btm)
    assert (lzb < K).any()                  # some convect
    assert (lcl < K).any()                  # some have a raised LCL (elevated layer)


def test_dt_tg_matches_fortran(fx, result):
    dt_tg = result[0]
    assert np.allclose(dt_tg, fx["dc_dt_tg"], rtol=1e-12, atol=1e-16)


def test_cape_cin_match_fortran(fx, result):
    _, cape, cin, _, _ = result
    assert np.allclose(cape, fx["dc_cape"], rtol=1e-11, atol=1e-9)
    assert np.allclose(cin, fx["dc_cin"], rtol=1e-11, atol=1e-9)


def test_lzb_lcl_indices_match_fortran(fx, result):
    _, _, _, lzb, lcl = result
    # Fortran indices are 1-based; jsca returns 0-based level indices.
    assert np.array_equal(lzb, fx["dc_lzb"].astype(int) - 1)
    assert np.array_equal(lcl, fx["dc_lcl"].astype(int) - 1)


def test_energy_conservation(fx, result):
    """The adjustment conserves column enthalpy: the mass-weighted mean temperature
    tendency over each column is ~0 (Isca shifts the parcel profile to enforce it)."""
    dt_tg = result[0]
    dp = fx["dc_phalf"][..., 1:] - fx["dc_phalf"][..., :-1]
    col_mean = np.sum(dt_tg * dp, axis=-1) / np.sum(dp, axis=-1)
    assert np.max(np.abs(col_mean)) < 1e-10   # K/s
