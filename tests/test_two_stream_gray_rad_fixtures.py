"""Tier-1 tests for Frierson grey radiation against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_two_stream_gray_rad_reference.F90``,
which compiles the **unmodified** ``two_stream_gray_rad.F90`` with the Frierson
namelist (``rad_scheme='frierson', do_seasonal=.false., atm_abs=0.2``), runs the
down + up passes over a latitude/pressure/temperature grid, and dumps the
model-facing outputs: the radiative heating profile ``tdt`` (passed in as 0, so
it returns ``tdt_rad``) and the surface downward SW/LW fluxes.

The scheme is pure arithmetic (``exp``/power laws, no lookup tables), so there is
no documented deviation: the fluxes match to machine precision and the heating
to the log/exp tolerance band (its small ``dp`` denominators amplify rounding
slightly, but the absolute error is ~1e-19 K/s).
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics import GrayRadParams, two_stream_gray_rad

FIXTURE = Path(__file__).parent / "fixtures" / "two_stream_gray_rad_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="two_stream_gray_rad fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _run(fx):
    tdt, net_sw, lw_dn, olr, net_lw = two_stream_gray_rad(
        GrayRadParams(), fx["rad_lat"], fx["rad_phalf"],
        fx["rad_t"], fx["rad_tsurf"], fx["rad_albedo"],
    )
    return np.asarray(tdt), np.asarray(net_sw), np.asarray(lw_dn)


def test_heating_matches_fortran(fx):
    tdt, _, _ = _run(fx)
    assert np.allclose(tdt, fx["rad_tdt"], rtol=1e-11, atol=1e-16)


def test_net_surface_sw_matches_fortran(fx):
    _, net_sw, _ = _run(fx)
    assert np.allclose(net_sw, fx["rad_net_sw_sfc"], rtol=1e-12, atol=1e-12)


def test_surface_lw_down_matches_fortran(fx):
    _, _, lw_dn = _run(fx)
    assert np.allclose(lw_dn, fx["rad_lw_down_sfc"], rtol=1e-12, atol=1e-12)


def test_physical_ranges(fx):
    """Sanity: surface fluxes positive; insolation peaks in the tropics."""
    _, net_sw, lw_dn = _run(fx)
    assert np.all(net_sw > 0) and np.all(lw_dn > 0)
    # equatorward latitudes receive more shortwave than the poles (p2 profile)
    lat = fx["rad_lat"][0]                       # (nlat,) — longitude-independent
    eq = np.argmin(np.abs(lat))
    assert net_sw[0, eq] > net_sw[0, 0]          # equator brighter than south pole
