"""Tier-1 tests for saturation vapor pressure against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_sat_vapor_pres_reference.F90``,
which calls Isca's unmodified ``sat_vapor_pres_k.F90`` kernel with
``do_simple=.true.`` (Frierson's setting) over a physical T,p sweep.

Documented deviation: Isca evaluates ``es`` (and its derivative) by quadratic
interpolation of a precomputed table (``lookup_es``/``compute_qs``); jsca
evaluates the closed form the table is *built* from. The residual is therefore
Isca's table-interpolation error, not a port error — jsca's direct form is the
more accurate of the two. Achieved: es/qs ~2e-7, dqs/dT ~5e-5 (the derivative
table is coarser); held at 1e-6 / 1e-4 relative.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics.sat_vapor_pres import (
    saturation_specific_humidity,
    saturation_specific_humidity_and_deriv,
    saturation_vapor_pressure,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sat_vapor_pres_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="sat_vapor_pres fixtures not generated"
)

ES_RTOL = 1e-6   # table-interpolation-limited
DQS_RTOL = 1e-4  # derivative table is coarser


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30))


def test_es_matches_fortran(fx):
    es = saturation_vapor_pressure(fx["svp_temp"])
    assert _relerr(es, fx["svp_es"]) < ES_RTOL


def test_qs_matches_fortran(fx):
    qs = saturation_specific_humidity(fx["svp_temp"], fx["svp_press"])
    assert _relerr(qs, fx["svp_qs"]) < ES_RTOL


def test_qs_and_dqsdt_matches_fortran(fx):
    qs, dqsdt = saturation_specific_humidity_and_deriv(fx["svp_temp"], fx["svp_press"])
    assert _relerr(qs, fx["svp_qs"]) < ES_RTOL
    assert _relerr(dqsdt, fx["svp_dqsdt"]) < DQS_RTOL
