"""Tier-1 tests for the boundary-layer diffusivity against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_diffusivity_reference.F90``,
which compiles the **unmodified** ``diffusivity.F90`` with the real monin_obukhov
kernel (for ``mo_diff``) and the Frierson namelist (``do_simple=.true.,
do_entrain=.false.``) over a column set spanning stable→unstable surface-layer
buoyancy scales. Dumps ``k_m``/``k_t`` (the momentum/heat eddy diffusivities on
the half levels) and ``h`` (the PBL depth).

Pure arithmetic (the ``mo_diff`` similarity functions are ``log``/powers, plus
the Richardson-number PBL walk), so there is no documented deviation: the
diffusivities match to machine precision and the PBL depth exactly.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics import DiffusivityParams, diffusivity

FIXTURE = Path(__file__).parent / "fixtures" / "diffusivity_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="diffusivity fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def result(fx):
    return [np.asarray(a) for a in diffusivity(
        DiffusivityParams(), fx["df_t"], np.zeros_like(fx["df_t"]),
        fx["df_u"], fx["df_v"], fx["df_z_full"], fx["df_z_half"],
        fx["df_u_star"], fx["df_b_star"])]


def test_k_m_matches_fortran(fx, result):
    k_m, _, _ = result
    assert np.allclose(k_m, fx["df_k_m"], rtol=1e-12, atol=1e-12)


def test_k_t_matches_fortran(fx, result):
    _, k_t, _ = result
    assert np.allclose(k_t, fx["df_k_t"], rtol=1e-12, atol=1e-12)


def test_pbl_depth_matches_fortran(fx, result):
    _, _, h = result
    assert np.allclose(h, fx["df_h"], rtol=1e-12, atol=1e-9)


def test_physical_structure(fx, result):
    """K vanishes above the PBL and peaks inside it; PBL depth is positive."""
    k_m, k_t, h = result
    assert np.all(h > 0)
    # top half level (k=0) is above the PBL -> zero diffusivity
    assert np.allclose(k_m[..., 0], 0.0)
    # some interior level carries non-trivial mixing
    assert k_m.max() > 1.0 and k_t.max() > 1.0
