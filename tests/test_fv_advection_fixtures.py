"""Tier-1 tests for fv_advection against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_fv_advection_reference.F90``,
which compiles Isca's actual ``fv_advection.F90`` unmodified (fms_mod stubbed;
mpp_domains_mod is the serial single-PE stub). Regeneration recipe in that
file's header.

Fortran grid storage is ``(lon, lat, level)``; the port uses ``(..., nlat,
nlon)`` with level a leading batch axis, so fixtures move axes to
``(level, lat, lon)``. Two cases are checked: the default advective form and the
``flux=True`` pure-flux form. The fixture winds push the near-pole zonal Courant
number above 1, so the ``integer_flux_x`` path is exercised (asserted below).
"""

from functools import partial
from pathlib import Path

import jax
import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.dycore import a_grid_horiz_advection, fv_advection_init

FIXTURE = Path(__file__).parent / "fixtures" / "fv_advection_reference.npz"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="fv_advection fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def to_port(a):
    """Fortran (lon, lat, level) -> port (level, lat, lon)."""
    return np.transpose(a, (2, 1, 0))


@pytest.fixture(scope="module")
def params(fx):
    nx = int(fx["fv_meta"][0])
    degrees_lon = float(fx["fv_meta"][4])
    return fv_advection_init(nx, fx["fv_yy"], degrees_lon)


@pytest.mark.parametrize("tag", ["adv", "flux"])
def test_fv_advection(fx, params, tag):
    dt = float(fx["fv_meta"][3])
    out = a_grid_horiz_advection(
        to_port(fx["fv_ua"]),
        to_port(fx["fv_va"]),
        to_port(fx["fv_q"]),
        dt,
        to_port(fx["fv_dqdt_in"]),
        params,
        flux=(tag == "flux"),
    )
    ref = to_port(fx[f"fv_dqdt_{tag}"])
    # pure-arithmetic kernel: absolute error is ~1e-17 on ~1e-3 fields; atol
    # covers the handful of near-cancellation points where |ref| ~ 1e-12.
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-14, atol=1e-16)


def test_jit_and_2d(fx, params):
    """jit (flux static) matches eager; the batch-free 2-D case matches level 0."""
    dt = float(fx["fv_meta"][3])
    ua, va, q = to_port(fx["fv_ua"]), to_port(fx["fv_va"]), to_port(fx["fv_q"])
    d0 = to_port(fx["fv_dqdt_in"])
    ref = to_port(fx["fv_dqdt_adv"])
    jitted = jax.jit(partial(a_grid_horiz_advection, flux=False))
    out = np.asarray(jitted(ua, va, q, dt, d0, params))
    np.testing.assert_allclose(out, ref, rtol=1e-14, atol=1e-16)
    out2d = np.asarray(a_grid_horiz_advection(ua[0], va[0], q[0], dt, d0[0], params))
    np.testing.assert_allclose(out2d, ref[0], rtol=1e-14, atol=1e-16)


def test_integer_flux_path_exercised(fx, params):
    """The fixture drives |zonal Courant| > 1 somewhere, exercising integer_flux_x."""
    dt = float(fx["fv_meta"][3])
    uc = 0.5 * (np.roll(to_port(fx["fv_ua"]), 1, axis=-1) + to_port(fx["fv_ua"]))
    b = uc * dt / (params.dx * np.asarray(params.c)[:, None])
    assert np.max(np.abs(b)) > 1.0
