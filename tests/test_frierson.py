"""Stability smoke test for the end-to-end Frierson moist model stepping.

Every kernel composed here is golden-fixture-validated against Isca; this test
gates the *stepping assembly* (physics -> spectral dynamics -> leapfrog -> tracer
advection + water correction -> slot roll) by integrating a resting, near-dry
aquaplanet for a handful of steps at low resolution and checking it stays finite,
physical, and stable:

* no NaN/Inf in any prognostic after the run;
* humidity stays non-negative (water_borrowing is off for Frierson, but the tracer
  scheme + physics must not drive it negative here);
* surface pressure and temperature stay in physical bands;
* the run is jit/scan-safe.

The machine-precision step fixture and the climatology-vs-Isca comparison (roadmap
item 11c) need a pinned Isca Frierson reference (a full Isca build).
"""
import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.model.frierson import build_frierson, initial_state, integrate, step

# T10 with the 25 Frierson levels: small but exercises the full pipeline.
pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def model():
    return build_frierson(num_fourier=10, dt=720.0)


@pytest.fixture(scope="module")
def state0(model):
    return initial_state(model, humidity=1.0e-3)


def _finite_all(state):
    vors, divs, ts, ln_ps, qg, t_surf = state
    return all(np.all(np.isfinite(np.asarray(x))) for x in (vors, divs, ts, ln_ps, qg, t_surf))


def test_single_step_runs(model, state0):
    s1 = step(model, state0, model.dt, model.wave_matrix_cold)  # cold-start forward step
    assert _finite_all(s1)


def test_short_integration_stable(model, state0):
    """A few leapfrog steps from rest stay finite and physical."""
    s = integrate(model, state0, n_steps=6, cold_start=True)
    assert _finite_all(s)
    *_, qg, t_surf = s
    qg = np.asarray(qg)
    t_surf = np.asarray(t_surf)
    # humidity non-negative and small (kg/kg), SST near its initial value
    assert qg.min() > -1e-8
    assert qg.max() < 0.1
    assert np.all(t_surf > 250.0) and np.all(t_surf < 320.0)


def test_surface_pressure_physical(model, state0):
    from jsca.model.frierson import _grid_from_spectral
    s = integrate(model, state0, n_steps=6, cold_start=True)
    vors, divs, ts, ln_ps, _, _ = s
    _, _, _, ps = _grid_from_spectral(model, vors, divs, ts, ln_ps, 1)
    ps = np.asarray(ps)
    assert np.all(ps > 0.9e5) and np.all(ps < 1.1e5)
