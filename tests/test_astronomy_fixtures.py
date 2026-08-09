"""Tier-1 tests for the astronomy module (diurnal_solar) against real-Fortran
fixtures.

Fixtures come from ``fortran_instrumentation/dump_astronomy_reference.F90``,
which compiles the **unmodified** ``astronomy.f90`` (default orbital params:
ecc=0, obliq=23.439 deg, per=102.932 deg, num_angles=3600) and calls
``diurnal_solar`` over a lat/lon grid for 16 cases — 4 orbital positions
(``time_since_ae`` = 0, pi/2, pi, 3pi/2) x 2 times of day (``gmt`` = 0, pi) x
{instantaneous, time-averaged over a 3-hour step}. It dumps ``cosz``,
``fracday`` and, per case, ``[gmt, time_since_ae, dt, rrsun]`` (``dt`` < 0 flags
the instantaneous case).

The scheme is pure trig, so jsca matches Isca to the log/exp tolerance band. The
nine day/night time-averaging cases (F90 L1274-1364) and the polar day/night
saturation of the half-day angle are all exercised by the latitude sweep and the
solstice orbital positions.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics.astronomy import AstronomyParams, build_orbit_angle, diurnal_solar

FIXTURE = Path(__file__).parent / "fixtures" / "astronomy_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="astronomy fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def orb():
    # default orbital params match the fixture driver's astronomy_nml defaults
    return AstronomyParams(), build_orbit_angle(AstronomyParams())


def _case(orb, fx, k):
    params, orb_angle = orb
    gmt, tsae, dt, _rrsun = fx["meta"][k]
    dt_arg = None if dt < 0 else dt
    cosz, fracday, rrsun = diurnal_solar(
        params, orb_angle, fx["lat"], fx["lon"], float(gmt), float(tsae), dt_arg)
    return np.asarray(cosz), np.asarray(fracday), float(rrsun)


def test_cosz_matches_fortran(fx, orb):
    for k in range(fx["meta"].shape[0]):
        cosz, _, _ = _case(orb, fx, k)
        assert np.allclose(cosz, fx["cosz"][k], rtol=1e-12, atol=1e-13), (
            f"cosz mismatch, case {k} (dt={fx['meta'][k, 2]:.3f})")


def test_fracday_matches_fortran(fx, orb):
    for k in range(fx["meta"].shape[0]):
        _, fracday, _ = _case(orb, fx, k)
        assert np.allclose(fracday, fx["fracday"][k], rtol=1e-12, atol=1e-13), (
            f"fracday mismatch, case {k}")


def test_rrsun_matches_fortran(fx, orb):
    # circular orbit (ecc=0) -> rrsun == 1 exactly at every orbital position
    for k in range(fx["meta"].shape[0]):
        _, _, rrsun = _case(orb, fx, k)
        assert np.isclose(rrsun, fx["meta"][k, 3], rtol=1e-13, atol=1e-13)


def test_physical_ranges(fx, orb):
    """cosz in [0,1], fracday in [0,1]; the sunlit hemisphere shifts with season."""
    for k in range(fx["meta"].shape[0]):
        cosz, fracday, _ = _case(orb, fx, k)
        assert np.all(cosz >= 0.0) and np.all(cosz <= 1.0 + 1e-12)
        assert np.all(fracday >= 0.0) and np.all(fracday <= 1.0 + 1e-12)

    lat = fx["lat"][:, 0]
    npole = np.argmax(lat)
    spole = np.argmin(lat)
    # instantaneous NH-summer solstice: time_since_ae ~ 3pi/2 (case index 12..).
    # Find the averaged case at tsae closest to 3pi/2 and check the daylit pole.
    tsae = fx["meta"][:, 1]
    k = int(np.argmin(np.abs(tsae - 1.5 * np.pi)))
    fracday = _case(orb, fx, k)[1]
    # at that orbital position one pole is in permanent day, the other in night
    assert fracday[npole].max() != fracday[spole].max()
