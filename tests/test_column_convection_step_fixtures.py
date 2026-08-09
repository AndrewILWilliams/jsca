"""Tier-1 golden test: the SCM's convection step vs a real Isca column step.

Unlike ``test_qe_moist_convection_fixtures.py`` (per-column inputs to the compiled
routine, Frierson namelist), this validates ``qe_moist_convection`` on the **exact
instantaneous input of a real Isca single-column run at the column_test config**
(rhbm=0.7, the 31 even-sigma levels), harvested by instrumenting
``idealized_moist_phys.F90`` to dump the convection I/O at step 600 (recipe:
``fortran_instrumentation/column_convection_step_recipe.md``).

Why this matters: the daily-mean jsca-vs-Isca precip differs by a few percent, and a
mean-state convection call over-rained by ~6% -- which *looked* like a convection
bug. It is not: fed Isca's true instantaneous step-600 column, jsca's convection
reproduces Isca's rain to +0.00%, its klzb / convflag exactly, and its T/q tendencies
to machine precision. So the precip difference is a *downstream* consequence of a
small boundary-layer profile difference, not a convection-scheme error. This test
locks that in.

Tolerances: the Betts-Miller reference profile is built from the saturation ``q_ref``,
so rain and **both** tendencies inherit the documented ``sat_vapor_pres`` es deviation
(~2e-7); all held at rtol 1e-6 (measured rel diffs: rain 3e-8, dT 2e-7, dq ~1e-8).
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics.qe_moist_convection import qe_moist_convection

FIXTURE = Path(__file__).parent / "fixtures" / "column_convection_step_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="column convection step fixture not generated")


@pytest.fixture(scope="module")
def result():
    fx = np.load(FIXTURE)
    dt = float(fx["qe_dt"][0])
    tin = fx["qe_tin"].astype(np.float64)[None, :]
    qin = fx["qe_qin"].astype(np.float64)[None, :]
    pf = fx["qe_pfull"].astype(np.float64)[None, :]
    ph = fx["qe_phalf"].astype(np.float64)[None, :]
    rain, dT, dq, cflag, _klcl = qe_moist_convection(tin, qin, pf, ph, dt)
    return fx, {"rain": np.asarray(rain).ravel(), "dT": np.asarray(dT).ravel(),
                "dq": np.asarray(dq).ravel(), "cflag": np.asarray(cflag).ravel()}


def test_temperature_tendency_match(result):
    fx, r = result
    assert np.allclose(r["dT"], fx["qe_dtg"], rtol=1e-6, atol=1e-8)


def test_rain_and_humidity_tendency_match(result):
    fx, r = result
    assert np.allclose(r["rain"], fx["qe_rain"], rtol=1e-6, atol=1e-12)
    assert np.allclose(r["dq"], fx["qe_dqg"], rtol=1e-6, atol=1e-11)


def test_convective_flag_matches(result):
    """Same convection branch (klzb / convflag) as Isca at the real column state."""
    fx, r = result
    assert float(r["cflag"][0]) == float(fx["qe_flag"][0])
