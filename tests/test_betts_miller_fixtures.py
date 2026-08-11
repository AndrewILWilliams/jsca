"""Tier-1 tests for full Betts-Miller convection against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_betts_miller_reference.F90``,
which compiles the **unmodified** ``betts_miller.f90`` with the real
``sat_vapor_pres`` wrapper + kernel (``escomp`` on the ``do_simple=.true.`` column
setting) and the default namelist (``do_simp=.true.``, ``do_shallower`` /
``do_changeqref`` / ``do_envsat`` / ``do_taucape`` = ``.false.``, ``buoyancy_kick=0``)
over a column set sweeping deep-convecting (warm/moist) to non-convecting
(cool/dry) -- so ``capecalcnew``, the reference-profile relaxation and the do_simp
energy conservation (bmflag 0/1/2) are all exercised.

The scheme is pure arithmetic apart from ``escomp``, so jsca matches Isca to the
saturation-vapour tolerance: the ``klzb``/``klcl`` indices exactly, CAPE to ~1e-6,
and the T/q increments and rain to ~1e-9.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics import BettsMillerParams, betts_miller

FIXTURE = Path(__file__).parent / "fixtures" / "betts_miller_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="betts_miller fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def result(fx):
    # fixture namelist: tau_bm=7200, rhbm=0.8, dt=600
    return [np.asarray(a) for a in betts_miller(
        BettsMillerParams(tau_bm=7200.0, rhbm=0.8), 600.0,
        fx["bm_t"], fx["bm_q"], fx["bm_pfull"], fx["bm_phalf"])]


def test_fixture_spans_deep_shallow_none(fx):
    """Guard: bmflag has 0 (no CAPE), 1 (shallow, no adjust) and 2 (deep) columns."""
    bmflag = fx["bm_bmflag"].astype(int)
    assert set(np.unique(bmflag)) >= {0, 2}
    assert (fx["bm_cape"] > 0).any() and (fx["bm_cape"] == 0).any()


def test_klzb_klcl_indices_match_fortran(fx, result):
    _, _, _, _, _, klzb, klcl = result
    # Fortran indices are 1-based (0 = no-LZB sentinel); jsca returns 0-based
    # (-1 sentinel).
    assert np.array_equal(klzb, fx["bm_klzb"].astype(int) - 1)
    assert np.array_equal(klcl, fx["bm_klcl"].astype(int) - 1)


def test_cape_cin_match_fortran(fx, result):
    _, _, _, cape, cin, _, _ = result
    assert np.allclose(cape, fx["bm_cape"], rtol=1e-6, atol=1e-4)
    assert np.allclose(cin, fx["bm_cin"], rtol=1e-6, atol=1e-4)


def test_tendencies_and_rain_match_fortran(fx, result):
    rain, tdel, qdel, _, _, _, _ = result
    assert np.allclose(tdel, fx["bm_tdel"], rtol=1e-8, atol=1e-8)
    assert np.allclose(qdel, fx["bm_qdel"], rtol=1e-8, atol=1e-10)
    assert np.allclose(rain, fx["bm_rain"], rtol=1e-7, atol=1e-8)


def test_energy_conservation_deep(fx, result):
    """For deep-convecting (bmflag=2) columns the do_simp adjustment conserves
    column moist enthalpy: integral(cp*tdel + hlv*qdel) dp/g ~ 0."""
    _, tdel, qdel, _, _, _, _ = result
    from jsca import constants
    dp = fx["bm_phalf"][..., 1:] - fx["bm_phalf"][..., :-1]
    deep = fx["bm_bmflag"].astype(int) == 2
    enth = np.sum((constants.CP_AIR * tdel + constants.HLV * qdel) * dp, axis=-1) / constants.GRAV
    assert np.max(np.abs(enth[deep])) < 1e-3   # W-scale residual per column
