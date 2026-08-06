"""Tier-1 tests for Monin-Obukhov similarity against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_monin_obukhov_reference.F90``,
which calls Isca's dependency-free ``monin_obukhov_kernel.F90`` (``mo_drag`` +
``mo_profile``) directly with the default namelist the Frierson aquaplanet uses
(``stable_option=1, rich_crit=2.0, neutral=.false., drag_min=1e-5``), over a
sweep of surface-layer states spanning the unstable / neutral / stable /
near-critical regimes so every stability branch and the Newton ``zeta`` solver
are exercised.

Pure arithmetic (``log``/``atan``/powers, no lookup tables), so there is no
documented deviation: the drag coefficients and friction scales match to machine
precision and the profile ratios exactly.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics import MOParams, mo_drag, mo_profile

FIXTURE = Path(__file__).parent / "fixtures" / "monin_obukhov_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="monin_obukhov fixtures not generated"
)

RTOL = 1e-12   # log/exp-bearing, machine-precision in practice


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _drag(fx):
    return [np.asarray(a) for a in mo_drag(
        MOParams(), fx["mo_pt"], fx["mo_pt0"], fx["mo_z"],
        fx["mo_z0"], fx["mo_zt"], fx["mo_zq"], fx["mo_speed"])]


def test_drag_coefficients_match_fortran(fx):
    drag_m, drag_t, drag_q, _, _ = _drag(fx)
    assert np.allclose(drag_m, fx["mo_drag_m"], rtol=RTOL, atol=1e-18)
    assert np.allclose(drag_t, fx["mo_drag_t"], rtol=RTOL, atol=1e-18)
    assert np.allclose(drag_q, fx["mo_drag_q"], rtol=RTOL, atol=1e-18)


def test_friction_scales_match_fortran(fx):
    _, _, _, u_star, b_star = _drag(fx)
    assert np.allclose(u_star, fx["mo_u_star"], rtol=RTOL, atol=1e-15)
    assert np.allclose(b_star, fx["mo_b_star"], rtol=RTOL, atol=1e-15)


def test_profile_ratios_match_fortran(fx):
    # chain the real path: mo_drag's friction scales into mo_profile
    _, _, _, u_star, b_star = _drag(fx)
    del_m, del_t, del_q = (np.asarray(a) for a in mo_profile(
        MOParams(), 10.0, 2.0, fx["mo_z"], fx["mo_z0"], fx["mo_zt"], fx["mo_zq"],
        u_star, b_star))
    assert np.allclose(del_m, fx["mo_del_m"], rtol=RTOL, atol=1e-12)
    assert np.allclose(del_t, fx["mo_del_t"], rtol=RTOL, atol=1e-12)
    assert np.allclose(del_q, fx["mo_del_q"], rtol=RTOL, atol=1e-12)


def test_regime_coverage_and_physics(fx):
    """The fixture spans unstable (large drag) to near-critical stable (drag_min)."""
    drag_m, _, _, _, _ = _drag(fx)
    assert np.isclose(drag_m.min(), MOParams().drag_min, rtol=1e-6)  # near-critical stable
    assert drag_m.max() > 1e-3                                       # strongly unstable
    # drag increases with instability (pt0 > pt): compare a matched wind pair
    unstable = fx["mo_pt0"] > fx["mo_pt"]
    stable = fx["mo_pt0"] < fx["mo_pt"]
    assert drag_m[unstable].max() > drag_m[stable].min()
