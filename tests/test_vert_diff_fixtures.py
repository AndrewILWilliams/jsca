"""Tier-1 tests for the implicit vertical diffusion against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_vert_diff_reference.F90``, which
compiles the **unmodified** ``vert_diff.F90`` and runs the Frierson down/up split
(``gcm_vert_diff_down`` then ``gcm_vert_diff_up``, ``do_conserve_energy=.true.,
do_virtual=.false.``, single ``sphum`` tracer) over a column set with K-profiles
and a surface stress. Dumps the momentum/temperature/humidity tendencies and the
frictional dissipative heating.

The up sweep here uses the down pass's ``Tri_surf`` directly (no mixed-layer
update in between — that is item 8), which is a well-defined diffusion with the
T/q surface flux held at its stored value. Pure arithmetic (tridiagonal Thomas
elimination), so there is no documented deviation: everything matches to machine
precision.

Note: the fixture dumps the **input** surface stress (``tau_u``/``tau_v`` are
``intent(inout)`` and overwritten by the call), so it feeds :func:`vert_diff_down`
correctly.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics import vert_diff_down, vert_diff_up

FIXTURE = Path(__file__).parent / "fixtures" / "vert_diff_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="vert_diff fixtures not generated"
)

DELT = 720.0


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def result(fx):
    dt_u, dt_v, diss, tri = vert_diff_down(
        DELT, fx["vd_u"], fx["vd_v"], fx["vd_t"], fx["vd_q"],
        fx["vd_diff_m"], fx["vd_diff_t"], fx["vd_p_half"], fx["vd_p_full"],
        fx["vd_z_full"], fx["vd_tau_u"], fx["vd_tau_v"],
        fx["vd_dtau_du"], fx["vd_dtau_dv"])
    dt_t, dt_q = vert_diff_up(DELT, tri)
    return {"dt_u": np.asarray(dt_u), "dt_v": np.asarray(dt_v),
            "dt_t": np.asarray(dt_t), "dt_q": np.asarray(dt_q),
            "diss": np.asarray(diss)}


@pytest.mark.parametrize("attr,key", [
    ("dt_u", "vd_dt_u"), ("dt_v", "vd_dt_v"), ("dt_t", "vd_dt_t"),
    ("dt_q", "vd_dt_q"), ("diss", "vd_diss_heat"),
])
def test_matches_fortran(result, fx, attr, key):
    assert np.allclose(result[attr], fx[key], rtol=1e-12, atol=1e-18)


def test_physical_structure(result, fx):
    """Diffusion acts near the surface (where K is large); dissipation heats."""
    # tendencies vanish in the free troposphere (top half of the column)
    K = fx["vd_u"].shape[-1]
    assert np.allclose(result["dt_u"][..., :K // 2], 0.0, atol=1e-12)
    # frictional dissipation is a heating (>= 0) wherever momentum is mixed
    assert np.all(result["diss"] >= -1e-18)
