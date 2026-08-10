"""Tests for the ``convection_scheme='NONE'`` (NO_CONV) option in the SCM.

NO_CONV (F90 ``idealized_moist_phys`` case ``NO_CONV``, L992-995) applies **no**
convective adjustment: zero convective T/q increments and no convective rain.
Large-scale condensation still runs, so precipitation and latent heating come
from ``lscale_cond`` alone. The scheme itself is therefore trivial; its
faithfulness rides on the already golden-fixture-validated components
(``lscale_cond``, ``two_stream_gray_rad``, ``surface_flux``, ``diffusivity`` on
the ``do_simple=.false.`` path, ``mixed_layer``, ``vert_diff``).

**Why there is no tight NO_CONV-vs-Isca trajectory test.** With convection off the
column is only marginally stratified near the surface (dry-static-energy uniform
to ~0.1 K across the PBL), so the boundary-layer *depth* -- a threshold crossing
-- is genuinely ill-conditioned: a ~1e-6 difference in the surface-flux
``u_star``/``b_star`` (jsca's ``surface_flux`` matches Isca to ~1e-6, not machine
precision) moves the PBL top by tens of metres, and over a 40-day integration of
the convectively-unstable column that amplifies into O(K) SST differences. Fed
*identical* surface fluxes the diffusivity is exact
(``test_diffusivity_nosimple_fixtures.py``, PBL depth to machine precision), and a
per-step comparison against an instrumented Isca NO_CONV run matches everywhere
except the ill-conditioned PBL levels -- so this is a property of the NO_CONV
configuration, not a port error. These tests therefore check that the option is
wired correctly and stable, not a golden trajectory.
"""
import jax
import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.model import column as C


@pytest.fixture(scope="module")
def noconv_model():
    # NO_CONV has no convective LCL, so it pairs with the bulk-Richardson PBL.
    return C.build_column(convection_scheme="NONE", do_lcl_diffusivity_depth=False)


def test_noconv_step_is_finite_and_physical(noconv_model):
    m = noconv_model
    s = jax.jit(lambda st: C.step(m, st, m.dt))(C.initial_state(m))
    u, v, tg, qg, ps, t_surf = s
    for a in (tg, qg, t_surf, ps):
        assert np.all(np.isfinite(np.asarray(a)))
    assert np.all(np.asarray(qg) > -1e-8)
    assert np.all(np.asarray(t_surf) > 200.0) and np.all(np.asarray(t_surf) < 360.0)


def test_noconv_runs_stably(noconv_model):
    m = noconv_model
    s = C.integrate(m, C.initial_state(m), n_steps=200, cold_start=True)
    _, _, tg, qg, _, t_surf = s
    assert np.all(np.isfinite(np.asarray(tg)))
    assert np.all(np.isfinite(np.asarray(qg)))
    assert np.all(np.asarray(t_surf) > 200.0) and np.all(np.asarray(t_surf) < 360.0)


def test_noconv_differs_from_betts_miller():
    """The scheme selection does something: turning convection off gives a
    genuinely different one-step state than the default Betts-Miller scheme."""
    m_none = C.build_column(convection_scheme="NONE", do_lcl_diffusivity_depth=False)
    m_bm = C.build_column()  # SIMPLE_BETTS_MILLER default
    s0 = C.initial_state(m_none)
    s_none = jax.jit(lambda st: C.step(m_none, st, m_none.dt))(s0)
    s_bm = jax.jit(lambda st: C.step(m_bm, st, m_bm.dt))(C.initial_state(m_bm))
    q_none = np.asarray(s_none[3][..., 1])
    q_bm = np.asarray(s_bm[3][..., 1])
    assert not np.allclose(q_none, q_bm)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="convection_scheme"):
        m = C.build_column(convection_scheme="RAS")
        jax.jit(lambda st: C.step(m, st, m.dt))(C.initial_state(m))
