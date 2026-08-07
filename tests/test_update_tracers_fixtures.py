"""Tier-1 tests for the grid tracer time-step against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_update_tracers_reference.F90``,
which compiles the **verbatim** grid branch of ``spectral_dynamics.F90``'s private
``update_tracers`` (lines 1224-1248) — calling the real, unmodified
``a_grid_horiz_advection`` and ``vert_advection`` — for the Frierson tracer
settings (one grid ``sphum`` tracer, ``robert_coeff=0.03``,
``hole_filling='off'``). It runs the **non-last** sub-step
(``step_number /= num_steps``), the branch that exercises the RAW dead-store quirk
(F90 L1243 overwritten by L1248).

The step assembly is scheme-agnostic, so the fixture uses ``second_centered``
vertical advection (a scheme jsca ports); Frierson's production scheme,
``finite_volume_parabolic`` (PPM), is the next vert_advection follow-up and slots
into the same :func:`update_grid_tracer` unchanged.

The horizontal advection carries the fv_advection documented deviation (its own
fixture holds it to ~1e-12), so the tracer levels match to ~1e-12, not machine
epsilon.

Fortran storage is ``(lon, lat, lev)``; the port uses jsca's ``(nlat, nlon, K)``
grid convention, so fixtures transpose lon<->lat.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401
from jsca.dycore import fv_advection_init, update_grid_tracer

FIXTURE = Path(__file__).parent / "fixtures" / "update_tracers_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="update_tracers fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _t3(a):
    """Fortran (lon, lat, lev) -> jsca (lat, lon, lev)."""
    return np.transpose(a, (1, 0, 2))


@pytest.fixture(scope="module")
def result(fx):
    nx = int(fx["ut_meta"][0])
    dt = float(fx["ut_meta"][3])
    fv = fv_advection_init(nx, np.asarray(fx["ut_yy"]), degrees_lon=360.0)
    q_cur_new, q_future, part_filt = update_grid_tracer(
        _t3(fx["ut_q_prev"]), _t3(fx["ut_q_cur_in"]), _t3(fx["ut_dt_tr"]),
        _t3(fx["ut_ug"]), _t3(fx["ut_vg"]), _t3(fx["ut_wg"]), _t3(fx["ut_p_half"]),
        dt, float(fx["ut_robert"][0]), float(fx["ut_raw"][0]), fv, last_step=False,
    )
    return {"q_cur": np.asarray(q_cur_new), "q_fut": np.asarray(q_future),
            "part_filt": np.asarray(part_filt)}


@pytest.mark.parametrize("attr,key", [
    ("q_cur", "ut_q_cur_out"),
    ("q_fut", "ut_q_fut_out"),
    ("part_filt", "ut_part_filt"),
])
def test_matches_fortran(result, fx, attr, key):
    np.testing.assert_allclose(result[attr], _t3(fx[key]), rtol=1e-11, atol=1e-13)


def test_future_is_advected_tracer(result, fx):
    """The RAW dead-store quirk: the future level equals the advected tracer tr
    (== the reported q_fut), independent of robert_coeff — verified by matching."""
    np.testing.assert_allclose(result["q_fut"], _t3(fx["ut_q_fut_out"]), rtol=1e-11, atol=1e-13)


def test_current_is_raw_filtered(result, fx):
    """The current level moved from its input by the Robert/RAW filter increment
    robert*part_filt*raw (non-zero where advection changed the tracer)."""
    moved = np.abs(result["q_cur"] - _t3(fx["ut_q_cur_in"]))
    assert np.max(moved) > 0.0
