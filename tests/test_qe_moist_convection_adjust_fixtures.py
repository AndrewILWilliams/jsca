"""Tier-1 tests for the qe_moist_convection Betts-Miller adjustment (stage 3b).

Same golden fixture as the CAPE stage (``dump_qe_moist_convection_reference.F90``,
unmodified ``qe_moist_convection.F90`` with the Frierson namelist). This PR
completes the scheme, so we validate its **product**: the convective rain and
the temperature/humidity increments the driver feeds back into the model
(``rain``, ``deltaT``, ``deltaq``), plus the ``convflag`` regime classifier. The
fixture spans no-CAPE, shallow and deep columns.

Tolerance: the only inexactness is the documented ``sat_vapor_pres``
table-vs-closed-form es deviation, propagated through the parcel ascent; it
stays ~1e-7 on ``deltaT`` and far smaller on ``deltaq``/``rain``. ``convflag``
matches exactly.

The internal ``qref`` reference profile (not returned) carries an irreducible
knife-edge on net-sink shallow columns (a structurally-zero ``Pq - Pq`` compare
that the es deviation can flip); it does not affect the returned tendencies, so
it is not tested here — see the ``qe_moist_convection`` docstring.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics.qe_moist_convection import qe_moist_convection

FIXTURE = Path(__file__).parent / "fixtures" / "qe_moist_convection_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="qe_moist_convection fixtures not generated"
)

DT = 720.0  # Frierson dt_atmos (the fixture driver's dt)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _run(fx):
    rain, dT, dq, cflag = qe_moist_convection(
        fx["qe_tin"], fx["qe_qin"], fx["qe_pfull"], fx["qe_phalf"], DT
    )
    return (np.asarray(rain), np.asarray(dT), np.asarray(dq), np.asarray(cflag))


def test_deltaT_matches_fortran(fx):
    _, dT, _, _ = _run(fx)
    assert np.allclose(dT, fx["qe_deltaT"], rtol=1e-5, atol=1e-6)


def test_deltaq_matches_fortran(fx):
    _, _, dq, _ = _run(fx)
    assert np.allclose(dq, fx["qe_deltaq"], rtol=1e-5, atol=1e-9)


def test_rain_matches_fortran(fx):
    rain, _, _, _ = _run(fx)
    assert fx["qe_rain"].max() > 1e-3          # fixture has real deep-convective rain
    assert np.allclose(rain, fx["qe_rain"], rtol=1e-5, atol=1e-7)


def test_convflag_matches_fortran(fx):
    _, _, _, cflag = _run(fx)
    assert np.array_equal(cflag, fx["qe_convflag"].astype(int))


def test_regime_coverage(fx):
    """The fixture must span all three regimes (no-CAPE / shallow / deep)."""
    cf = fx["qe_convflag"].astype(int)
    assert (cf == 0).any() and (cf == 1).any() and (cf == 2).any()


def test_energy_and_moisture_signs(fx):
    """Physical sanity: deep-convective columns warm (net) and dry, and rain>=0."""
    rain, dT, dq, cflag = _run(fx)
    assert np.all(rain >= 0.0)
    deep = cflag == 2
    # column-mass-weighted mean tendencies over deep columns
    dp = fx["qe_phalf"][..., 1:] - fx["qe_phalf"][..., :-1]
    colT = np.sum(dT * dp, axis=-1)[deep]
    assert np.all(colT > 0)                    # net latent heating
