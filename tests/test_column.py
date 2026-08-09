"""Stability / invariant smoke test for the single-column model (SCM) assembly.

Every physics kernel the SCM composes is golden-fixture-validated against the real
Fortran, and the leapfrog is the exact port of ``leapfrog_3d_real``; this test gates
the *driver assembly* (``jsca.model.column``) — the initial condition, the
fixed-wind/fixed-ps stepping, the ``q_decrease_only`` clamp, and jit/scan safety —
the way ``tests/test_frierson.py`` gates the full-model stepping.

Checks, over a short integration of a single Frierson-physics column:

* no NaN/Inf in any prognostic;
* **winds and surface pressure stay bit-for-bit fixed** (the defining SCM property —
  the momentum tendency is discarded and there is no continuity equation);
* humidity stays non-negative and, with ``q_decrease_only``, non-increasing with
  height;
* the slab SST stays in a physical band;
* the initial condition matches ``column_initialize_fields.F90`` arithmetic;
* the driver is jit/scan-safe.

A machine-precision golden column-step fixture from an instrumented Isca SCM run is
the remaining validation and needs a full Isca build (see the module docstring of
``jsca.model.column`` and ``fortran_instrumentation/dump_column_init_reference.F90``).
"""
import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca import constants
from jsca.dycore.leapfrog import apply_q_decrease_only
from jsca.model.column import GLOBAL_AVERAGE_LAT_DEG, build_column, initial_state, integrate

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def model():
    # single Frierson-physics column at the global-average latitude
    return build_column(dt=1440.0)


@pytest.fixture(scope="module")
def state0(model):
    return initial_state(model)


def _finite_all(state):
    return all(np.all(np.isfinite(np.asarray(x))) for x in state)


def test_initial_condition_matches_fortran(model, state0):
    """``column_initialize_fields.F90`` L74-79: surface winds, uniform T, hydrostatic
    ps from the surface geopotential."""
    u, v, tg, qg, ps, t_surf = state0
    u, v, ps = np.asarray(u), np.asarray(v), np.asarray(ps)
    # winds: surface_wind / sqrt(2) at the bottom level, zero aloft
    assert np.allclose(u[..., -1], 5.0 / np.sqrt(2.0))
    assert np.allclose(v[..., -1], 5.0 / np.sqrt(2.0))
    assert np.allclose(u[..., :-1], 0.0) and np.allclose(v[..., :-1], 0.0)
    # ps = exp(ln(p_ref) - Phi_s/(Rd T0)); Phi_s = 0 here -> ps = p_ref
    ln_ps = np.log(101325.0) - np.asarray(model.surf_geopotential) / (
        constants.RDGAS * 264.0)
    assert np.allclose(ps, np.exp(ln_ps))
    # two identical time levels (cold start: previous == current)
    assert np.array_equal(np.asarray(tg)[..., 0], np.asarray(tg)[..., 1])
    assert np.array_equal(np.asarray(qg)[..., 0], np.asarray(qg)[..., 1])
    # global-average latitude solves 3 sin^2(lat) - 1 = 0
    assert np.isclose(3.0 * np.sin(np.deg2rad(GLOBAL_AVERAGE_LAT_DEG)) ** 2 - 1.0, 0.0)


def test_short_integration_stable_and_invariants(model, state0):
    u0, v0, _, _, ps0, _ = state0
    s = integrate(model, state0, n_steps=30, cold_start=True)
    assert _finite_all(s)
    u, v, tg, qg, ps, t_surf = s

    # the defining SCM property: winds and surface pressure are held exactly fixed
    assert np.array_equal(np.asarray(u), np.asarray(u0))
    assert np.array_equal(np.asarray(v), np.asarray(v0))
    assert np.array_equal(np.asarray(ps), np.asarray(ps0))

    qg = np.asarray(qg)
    assert qg.min() > -1e-12                     # humidity non-negative
    assert qg.max() < 0.1                        # and small (kg/kg)
    # q_decrease_only: q non-increasing with height (k=0 top .. K-1 surface)
    q_cur = qg[..., 1]
    assert np.all(q_cur[..., :-1] <= q_cur[..., 1:] + 1e-15)
    # slab SST stays physical and near its start (2.5 m mixed layer, few steps)
    t_surf = np.asarray(t_surf)
    assert np.all(t_surf > 250.0) and np.all(t_surf < 320.0)


def test_temperature_actually_evolves(model, state0):
    """Sanity: the physics drives a non-trivial temperature tendency (the SCM is not
    a no-op) while winds/ps are frozen."""
    s = integrate(model, state0, n_steps=20, cold_start=True)
    t0 = np.asarray(state0[2])[..., 1]
    t1 = np.asarray(s[2])[..., 1]
    assert np.max(np.abs(t1 - t0)) > 1e-3


def test_q_decrease_only_clamp_is_reverse_cummin():
    """``apply_q_decrease_only`` reproduces the ``column.F90`` L782-792 upward sweep:
    each level is capped at the minimum of itself and everything below it."""
    rng = np.random.default_rng(0)
    q = rng.random((3, 2, 6))                      # (nlat, nlon, K)
    out = np.asarray(apply_q_decrease_only(q))
    expect = np.flip(np.minimum.accumulate(np.flip(q, axis=-1), axis=-1), axis=-1)
    assert np.allclose(out, expect)
    # result is non-increasing with height at every column
    assert np.all(out[..., :-1] <= out[..., 1:] + 1e-15)


def test_multi_column_independent(model):
    """Several columns run at once; a warmer-insolation (equatorward) column ends up
    with a warmer slab than a poleward one — the columns are independent but each
    responds to its own latitude."""
    m = build_column(latitudes=np.array([0.0, 60.0]), dt=1440.0)
    s0 = initial_state(m)
    s = integrate(m, s0, n_steps=40, cold_start=True)
    assert _finite_all(s)
    t_surf = np.asarray(s[5])[:, 0]
    assert t_surf[0] > t_surf[1]                   # equator warmer than 60 deg
