"""Tier-1 tests for the bulk surface fluxes against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_surface_flux_reference.F90``,
which compiles the **unmodified** ``surface_flux.F90`` together with the real
monin_obukhov and sat_vapor_pres modules, with the Frierson namelist
(``do_simple=.true., use_virtual_temp=.false., old_dtaudv=.true.``) over an
ocean point set (``land=F, seawater=T``, no bucket) spanning unstable→stable
air-sea contrasts and a range of winds.

The momentum/heat quantities are exact to machine precision. The moisture
quantities (``flux_q``, ``q_star``) carry the documented ``sat_vapor_pres``
table-vs-closed-form ``es`` deviation (~2e-7), which enters through the surface
saturation humidity ``q_sat``; ``dedt_surf`` (a finite difference of ``es``)
and ``rh_2m`` inherit it too. Held at 1e-6.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics import surface_flux

FIXTURE = Path(__file__).parent / "fixtures" / "surface_flux_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="surface_flux fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def result(fx):
    n = fx["sf_t_atm"].shape[0]
    return surface_flux(
        fx["sf_t_atm"], fx["sf_q_atm"], fx["sf_u_atm"], fx["sf_v_atm"],
        fx["sf_p_atm"], fx["sf_z_atm"], fx["sf_p_surf"], fx["sf_t_surf"],
        np.zeros(n), np.zeros(n),
        fx["sf_rough_mom"], fx["sf_rough_heat"], fx["sf_rough_moist"],
        fx["sf_gust"], fx["sf_q_surf_in"],
    )


# quantities that are exact (no es dependence)
_EXACT = [
    ("flux_t", "sf_flux_t"), ("flux_r", "sf_flux_r"), ("flux_u", "sf_flux_u"),
    ("flux_v", "sf_flux_v"), ("cd_m", "sf_cd_m"), ("cd_t", "sf_cd_t"),
    ("cd_q", "sf_cd_q"), ("w_atm", "sf_w_atm"), ("u_star", "sf_u_star"),
    ("b_star", "sf_b_star"), ("dhdt_surf", "sf_dhdt_surf"),
    ("dhdt_atm", "sf_dhdt_atm"), ("dedq_atm", "sf_dedq_atm"),
    ("drdt_surf", "sf_drdt_surf"), ("dtaudu_atm", "sf_dtaudu_atm"),
    ("temp_2m", "sf_temp_2m"), ("u_10m", "sf_u_10m"),
]
# quantities carrying the sat_vapor_pres es deviation
_ES = [("flux_q", "sf_flux_q"), ("q_star", "sf_q_star"),
       ("dedt_surf", "sf_dedt_surf"), ("q_2m", "sf_q_2m"), ("rh_2m", "sf_rh_2m")]


@pytest.mark.parametrize("attr,key", _EXACT)
def test_exact_quantities_match_fortran(result, fx, attr, key):
    assert np.allclose(np.asarray(getattr(result, attr)), fx[key],
                       rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("attr,key", _ES)
def test_moisture_quantities_match_fortran(result, fx, attr, key):
    assert np.allclose(np.asarray(getattr(result, attr)), fx[key],
                       rtol=1e-6, atol=1e-12)


def test_physical_signs(result, fx):
    """Sanity: flux_t > 0 where the ocean is warmer than the air, and < 0 where cooler."""
    flux_t = np.asarray(result.flux_t)
    warmer = fx["sf_t_surf"] > fx["sf_t_atm"]
    assert np.all(flux_t[warmer] > 0) and np.all(flux_t[~warmer] < 0)
    assert np.all(np.asarray(result.flux_r) > 0)          # upward LW positive


# --- use_virtual_temp = True path (column_test config) ---
# Fixture from dump_surface_flux_vt_reference.F90 (identical inputs, the namelist
# flipped to use_virtual_temp=.true.): the Monin-Obukhov drag then sees virtual
# potential temperatures and rho uses the virtual temperature.
FIXTURE_VT = Path(__file__).parent / "fixtures" / "surface_flux_vt_reference.npz"


@pytest.fixture(scope="module")
def fx_vt():
    if not FIXTURE_VT.exists():
        pytest.skip("surface_flux use_virtual_temp fixture not generated")
    return np.load(FIXTURE_VT)


@pytest.fixture(scope="module")
def result_vt(fx_vt):
    n = fx_vt["sf_t_atm"].shape[0]
    return surface_flux(
        fx_vt["sf_t_atm"], fx_vt["sf_q_atm"], fx_vt["sf_u_atm"], fx_vt["sf_v_atm"],
        fx_vt["sf_p_atm"], fx_vt["sf_z_atm"], fx_vt["sf_p_surf"], fx_vt["sf_t_surf"],
        np.zeros(n), np.zeros(n),
        fx_vt["sf_rough_mom"], fx_vt["sf_rough_heat"], fx_vt["sf_rough_moist"],
        fx_vt["sf_gust"], fx_vt["sf_q_surf_in"],
        use_virtual_temp=True,
    )


# With use_virtual_temp=True the surface virtual temperature thv_surf =
# t_surf*(1 + d608*q_sat) makes the Monin-Obukhov drag itself depend on q_sat, so
# the documented sat_vapor_pres es deviation (~2e-7) propagates (attenuated to
# ~1e-8 through the drag) into every drag-derived quantity -- not just the moisture
# outputs. So all quantities here are held at the es-deviation tolerance, not machine
# epsilon. flux_r / drdt_surf (pure t_surf^4) stay exact but pass the same bound.
@pytest.mark.parametrize("attr,key", _EXACT + _ES)
def test_vt_quantities_match_fortran(result_vt, fx_vt, attr, key):
    assert np.allclose(np.asarray(getattr(result_vt, attr)), fx_vt[key],
                       rtol=1e-6, atol=1e-12)


def test_vt_actually_differs_from_non_virtual(fx_vt):
    """The virtual-temp path must move the drag/fluxes vs the d608=0 path (same inputs),
    else the toggle would be a no-op."""
    n = fx_vt["sf_t_atm"].shape[0]
    common = (fx_vt["sf_t_atm"], fx_vt["sf_q_atm"], fx_vt["sf_u_atm"], fx_vt["sf_v_atm"],
              fx_vt["sf_p_atm"], fx_vt["sf_z_atm"], fx_vt["sf_p_surf"], fx_vt["sf_t_surf"],
              np.zeros(n), np.zeros(n), fx_vt["sf_rough_mom"], fx_vt["sf_rough_heat"],
              fx_vt["sf_rough_moist"], fx_vt["sf_gust"], fx_vt["sf_q_surf_in"])
    off = surface_flux(*common, use_virtual_temp=False)
    assert not np.allclose(np.asarray(off.flux_t), fx_vt["sf_flux_t"], rtol=1e-6)
