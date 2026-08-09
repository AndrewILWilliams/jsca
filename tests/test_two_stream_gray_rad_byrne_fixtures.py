"""Tier-1 tests for the Byrne & O'Gorman (2013) grey-radiation longwave scheme
against real-Fortran fixtures.

Fixtures come from
``fortran_instrumentation/dump_two_stream_gray_rad_byrne_reference.F90``, which
compiles the **unmodified** ``two_stream_gray_rad.F90`` with ``rad_scheme='byrne',
do_seasonal=.false.`` and runs the down + up passes over a latitude / pressure /
temperature / **humidity** grid (Byrne's LW optical depth is humidity-dependent,
so the driver uses a physically decreasing-with-height q profile). It dumps the
radiative heating ``tdt`` (passed in as 0, so it returns ``tdt_rad``) and the
surface downward SW/LW fluxes.

Byrne shares Frierson's shortwave and the same two-stream LW integration; only
the layer transmissivity differs (``dtau = (bog_a*bog_mu + 0.17*ln(CO2/360) +
bog_b*q)*dp/pstd_earth``). It is pure arithmetic (``exp``/``log``), so the fluxes
match to machine precision and the heating to the log/exp tolerance band.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics import GrayRadParams, two_stream_gray_rad

FIXTURE = Path(__file__).parent / "fixtures" / "two_stream_gray_rad_byrne_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="Byrne grey-radiation fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _run(fx):
    # rad_scheme="byrne" requires q; the driver namelist used atm_abs=0.2.
    params = GrayRadParams(rad_scheme="byrne", atm_abs=0.2)
    tdt, net_sw, lw_dn, olr, net_lw = two_stream_gray_rad(
        params, fx["rad_lat"], fx["rad_phalf"],
        fx["rad_t"], fx["rad_tsurf"], fx["rad_albedo"], fx["rad_q"],
    )
    return np.asarray(tdt), np.asarray(net_sw), np.asarray(lw_dn)


def test_heating_matches_fortran(fx):
    tdt, _, _ = _run(fx)
    assert np.allclose(tdt, fx["rad_tdt"], rtol=1e-11, atol=1e-16)


def test_net_surface_sw_matches_fortran(fx):
    # SW is identical to Frierson, so this also checks the shared branch is intact.
    _, net_sw, _ = _run(fx)
    assert np.allclose(net_sw, fx["rad_net_sw_sfc"], rtol=1e-12, atol=1e-12)


def test_surface_lw_down_matches_fortran(fx):
    _, _, lw_dn = _run(fx)
    assert np.allclose(lw_dn, fx["rad_lw_down_sfc"], rtol=1e-12, atol=1e-12)


def test_byrne_differs_from_frierson(fx):
    """Guard: the Byrne LW genuinely differs from Frierson on this state (so the
    scheme selection is doing something, not silently falling through)."""
    tdt_b, _, lw_b = _run(fx)
    tdt_f, _, lw_f, *_ = two_stream_gray_rad(
        GrayRadParams(rad_scheme="frierson"), fx["rad_lat"], fx["rad_phalf"],
        fx["rad_t"], fx["rad_tsurf"], fx["rad_albedo"],
    )
    assert not np.allclose(np.asarray(lw_b), np.asarray(lw_f), rtol=1e-3)


def test_physical_ranges(fx):
    """Sanity: surface fluxes positive; downward LW largest in the moist tropics
    (Byrne's humidity-dependent optical depth traps more LW where q is high)."""
    _, net_sw, lw_dn = _run(fx)
    assert np.all(net_sw > 0) and np.all(lw_dn > 0)
    lat = fx["rad_lat"][0]                        # (nlat,) — longitude-independent
    eq = np.argmin(np.abs(lat))
    assert lw_dn[0, eq] > lw_dn[0, 0]             # moist equator vs dry pole
