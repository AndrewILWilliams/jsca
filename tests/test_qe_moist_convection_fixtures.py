"""Tier-1 tests for the qe_moist_convection CAPE stage vs real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_qe_moist_convection_reference.F90``,
which compiles the **unmodified** ``qe_moist_convection.F90`` (+ the real
sat_vapor_pres wrapper) with the Frierson namelist (``rhbm=0.7, Tmin=160,
Tmax=350``) and dumps the full public output over a column set spanning the
scheme's regimes (no-CAPE, shallow, deep). This PR ports the CAPE-diagnosis
half, so we validate ``CAPE``, ``CIN`` and the LCL/LZB levels (the public
``cape``, ``cin``, ``klcl``, ``klzb`` outputs); the Betts-Miller adjustment
(``deltaT``/``deltaq``/``rain``) is a follow-up module.

Index convention: jsca returns 0-based ``kLZB``/``kLCL`` (``kLZB=0`` = no CAPE);
Isca's are 1-based, so the fixture values equal ``klcl+1`` and (where nonzero)
``klzb+1``.

Tolerance: the only inexactness is the documented ``sat_vapor_pres``
table-vs-closed-form es deviation (Isca's ``escomp`` interpolates a table; jsca
evaluates the closed form). Propagated through the iterated moist ascent it stays
~1e-7 relative on CAPE; held at 1e-5. The LCL/LZB integer levels match exactly.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics import convective_cape

FIXTURE = Path(__file__).parent / "fixtures" / "qe_moist_convection_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="qe_moist_convection fixtures not generated"
)

CAPE_RTOL = 1e-5   # sat_vapor_pres table deviation, propagated through the ascent
CIN_ATOL = 1e-3    # CIN is O(10-200 J/kg); abs error is ~1e-6


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _run(fx):
    cape, cin, klzb, klcl = convective_cape(
        fx["qe_tin"], fx["qe_qin"], fx["qe_pfull"], fx["qe_phalf"]
    )
    return (np.asarray(cape), np.asarray(cin),
            np.asarray(klzb), np.asarray(klcl))


def test_cape_matches_fortran(fx):
    cape, _, _, _ = _run(fx)
    rel = np.max(np.abs(cape - fx["qe_cape"]) / np.maximum(np.abs(fx["qe_cape"]), 1e-9))
    assert rel < CAPE_RTOL


def test_cin_matches_fortran(fx):
    _, cin, _, _ = _run(fx)
    assert np.allclose(cin, fx["qe_cin"], rtol=1e-5, atol=CIN_ATOL)


def test_lcl_level_matches_fortran(fx):
    _, _, _, klcl = _run(fx)
    # jsca 0-based -> Isca 1-based
    assert np.array_equal(klcl + 1, fx["qe_klcl"].astype(int))


def test_lzb_level_matches_fortran(fx):
    _, _, klzb, _ = _run(fx)
    klzb_f = np.where(klzb > 0, klzb + 1, 0)   # kLZB=0 is the "no CAPE" sentinel
    assert np.array_equal(klzb_f, fx["qe_klzb"].astype(int))


def test_regime_coverage(fx):
    """Guard that the fixture spans convecting and non-convecting columns."""
    cape, _, klzb, _ = _run(fx)
    assert (cape > 0).sum() > 0        # some columns have CAPE
    assert (cape == 0).sum() > 0       # some columns have none
    assert (klzb > 0).sum() > 0        # some columns reach a level of zero buoyancy
