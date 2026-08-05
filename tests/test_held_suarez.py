"""End-to-end smoke tests for the Held-Suarez model.

The full HS94 climatology (eddy-driven jets) needs a multi-week integration, too
slow for CI. These tests instead check that the assembled model *integrates*: a
from-rest state stays finite and bounded, spins up (kinetic energy grows from
zero under the forcing), and — crucially — conserves global dry-air mass to
machine precision via the mass correction. They also exercise the ``lax.scan``
integration path (jit-compiled step).
"""

import jax.numpy as jnp
import numpy as np

import jsca  # noqa: F401
from jsca.grid.transforms import area_weighted_global_mean
from jsca.model import build_held_suarez, initial_state, integrate
from jsca.model.held_suarez import _grid_from_spectral


def _diagnostics(m, state):
    u, v, _t, ps = _grid_from_spectral(m, *state, 1)
    ke = float(area_weighted_global_mean(m.dyn.transforms, jnp.mean(0.5 * (u**2 + v**2), -1)))
    mean_ps = float(area_weighted_global_mean(m.dyn.transforms, ps))
    finite = bool(np.isfinite(np.asarray(u)).all() and np.isfinite(np.asarray(ps)).all())
    return ke, mean_ps, finite


def test_integrates_stably_and_conserves_mass():
    m = build_held_suarez(num_fourier=8, num_levels=8, dt=1200.0)
    st = initial_state(m, temperature=264.0)
    ke0, mps0, finite0 = _diagnostics(m, st)
    assert ke0 == 0.0 and finite0  # starts at rest

    st = integrate(m, st, 30)  # jit + lax.scan
    ke1, mps1, finite1 = _diagnostics(m, st)

    assert finite1  # no blow-up
    assert ke1 > ke0  # spins up from rest under HS forcing
    assert ke1 < 1.0e4  # stays bounded (physical KE, not a numerical explosion)
    # global dry-air mass (mean surface pressure) held by the mass correction
    np.testing.assert_allclose(mps1, mps0, rtol=1e-8)


def test_sampling_path():
    """The climatology-accumulation path (sample_every) returns snapshots."""
    m = build_held_suarez(num_fourier=8, num_levels=8, dt=1200.0)
    st = initial_state(m)
    st, samples = integrate(m, st, 20, sample_every=10)
    u, t, ps = samples
    assert u.shape == (2, m.nlat, m.nlon, m.dyn.num_levels)  # 2 snapshots
    assert np.isfinite(np.asarray(u)).all()
