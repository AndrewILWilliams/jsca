"""Tier-1 tests for the bucket path of surface_flux against real-Fortran fixtures.

The bucket model (Manabe-style) makes land evaporation depend on soil moisture. In
``surface_flux`` (F90 ``src/coupler/surface_flux.F90`` L448-643) this means: a dry
surface (empty bucket) has ``q_surf0 = q_atm`` so there is no humidity gradient and
no evaporation; over land the evaporation and its ``dedt_surf`` derivative are
ramped by ``bucket_depth/(0.75*max)`` once the bucket drops below 0.75 of its
capacity; and the latent-heat depletion of the bucket (``depth_change_lh``) is
capped at the water actually available.

Fixture from ``fortran_instrumentation/dump_surface_flux_bucket_reference.F90``
(the unmodified ``surface_flux.F90`` with ``bucket=.true.``, 30 land points whose
``bucket_depth`` sweeps 0 -> 2 across the 0.75*max = 1.5 threshold, plus 10 ocean
points). jsca matches Isca to the saturation-vapour tolerance.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics.monin_obukhov import MOParams
from jsca.physics.surface_flux import surface_flux

FIXTURE = Path(__file__).parent / "fixtures" / "surface_flux_bucket_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="surface_flux bucket fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def result(fx):
    n = fx["sf_t_atm"].shape[0]
    land = fx["sf_land"] > 0.5
    # fixture namelist: do_simple=.true., use_virtual_temp=.false.; dt=720, max=2.0
    return surface_flux(
        fx["sf_t_atm"], fx["sf_q_atm"], fx["sf_u_atm"], fx["sf_v_atm"],
        fx["sf_p_atm"], fx["sf_z_atm"], fx["sf_p_surf"], fx["sf_t_surf"],
        np.zeros(n), np.zeros(n),
        fx["sf_rough_mom"], fx["sf_rough_heat"], fx["sf_rough_moist"], fx["sf_gust"],
        fx["sf_q_surf_in"], MOParams(), False,
        bucket=True, bucket_depth=fx["sf_bucket_depth"], max_bucket_depth_land=2.0,
        land=land, dt=720.0)


def test_fixture_spans_empty_to_full_land(fx):
    """Guard: land bucket sweeps empty, below-threshold (beta ramp) and full."""
    land = fx["sf_land"] > 0.5
    bd = fx["sf_bucket_depth"][land]
    assert (bd <= 0.0).any() and ((bd > 0) & (bd < 1.5)).any() and (bd >= 1.5).any()


def test_flux_q_matches_fortran(fx, result):
    assert np.allclose(np.asarray(result.flux_q), fx["sf_flux_q"], rtol=1e-7, atol=1e-12)


def test_dedt_surf_matches_fortran(fx, result):
    assert np.allclose(np.asarray(result.dedt_surf), fx["sf_dedt_surf"], rtol=1e-7, atol=1e-12)


def test_depth_change_lh_matches_fortran(fx, result):
    assert np.allclose(np.asarray(result.depth_change_lh), fx["sf_depth_change_lh"],
                       rtol=1e-7, atol=1e-12)


def test_empty_bucket_kills_evaporation(fx, result):
    """A land point with an empty bucket has exactly zero evaporation."""
    land = fx["sf_land"] > 0.5
    empty = land & (fx["sf_bucket_depth"] <= 0.0)
    assert np.all(np.asarray(result.flux_q)[empty] == 0.0)
    assert np.all(np.asarray(result.depth_change_lh)[empty] == 0.0)
