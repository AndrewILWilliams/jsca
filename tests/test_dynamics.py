"""Validation for the assembled dry dynamical-core tendency computation.

``compute_tendencies`` composes the fixture-validated kernels (four_in_one,
compute_pressure_gradient, vertical/horizontal advection, the wind<->vor/div
transforms, implicit_correction, spectral_damping) into the per-step spectral
tendencies of Isca's ``spectral_dynamics`` (F90 L845-910).

The whole assembly is pinned by a physical invariant that no single kernel can
satisfy on its own: **a resting, isothermal atmosphere is an exact steady state**
of the adiabatic dynamics, so every tendency must be identically zero. A
mis-placed transpose (the level-last <-> level-leading juggling), a wrong sign in
the vorticity flux, or a mis-wired implicit/damping call all break it. A
perturbed state must instead produce finite, bounded tendencies, and the whole
thing must be jit-safe.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jsca  # noqa: F401
from jsca.dycore import build_dynamics_params, compute_tendencies, rest_state
from jsca.dycore.implicit import build_wave_matrices

NF = 10
NLAT = 2 * NF + 2
NLON = 4 * NF + 4
K = 8
DT = 600.0


@pytest.fixture(scope="module")
def setup():
    p = build_dynamics_params(NF, NLAT, NLON, K)
    wm = build_wave_matrices(p.implicit, DT)
    return p, wm


def _stack(vor, div, ts, lnps):
    """Two identical time levels (previous == current)."""
    return (
        jnp.stack([vor, vor], -1),
        jnp.stack([div, div], -1),
        jnp.stack([ts, ts], -1),
        jnp.stack([lnps, lnps], -1),
    )


def test_rest_state_is_steady(setup):
    """Resting isothermal atmosphere -> all tendencies identically zero."""
    p, wm = setup
    vor, div, ts, lnps = rest_state(p, NLAT, NLON, temperature=280.0, surface_press=1.0e5)
    vors, divs, tss, lnpss = _stack(vor, div, ts, lnps)
    dvor, ddiv, dts, dlnps = compute_tendencies(p, vors, divs, tss, lnpss, DT, wm, 0, 1)
    # a real bug produces O(1e-4)..O(1) tendencies; machine zero is < 1e-11
    assert float(jnp.abs(dvor).max()) < 1e-11
    assert float(jnp.abs(ddiv).max()) < 1e-11
    assert float(jnp.abs(dts).max()) < 1e-11
    assert float(jnp.abs(dlnps).max()) < 1e-11


def test_perturbation_is_bounded(setup):
    """A warm mid-atmosphere temperature perturbation gives finite, bounded
    tendencies (and a nonzero response — the core actually does something)."""
    p, wm = setup
    vor, div, ts, lnps = rest_state(p, NLAT, NLON, temperature=280.0, surface_press=1.0e5)
    # add a small temperature perturbation in a single low-order spectral mode
    ts = ts.at[1, 1, K // 2].add(1.0 + 0.0j)
    vors, divs, tss, lnpss = _stack(vor, div, ts, lnps)
    dvor, ddiv, dts, dlnps = compute_tendencies(p, vors, divs, tss, lnpss, DT, wm, 0, 1)
    for x in (dvor, ddiv, dts, dlnps):
        assert np.all(np.isfinite(np.asarray(x)))
    # the perturbation drives a divergence/temperature response
    assert float(jnp.abs(ddiv).max()) > 0.0
    assert float(jnp.abs(dts).max()) > 0.0


def test_jit(setup):
    p, wm = setup
    vor, div, ts, lnps = rest_state(p, NLAT, NLON, temperature=280.0, surface_press=1.0e5)
    ts = ts.at[1, 1, K // 2].add(1.0 + 0.0j)
    vors, divs, tss, lnpss = _stack(vor, div, ts, lnps)
    eager = compute_tendencies(p, vors, divs, tss, lnpss, DT, wm, 0, 1)
    jitted = jax.jit(lambda *a: compute_tendencies(p, *a, 0, 1))(vors, divs, tss, lnpss, DT, wm)
    for e, j in zip(eager, jitted):
        np.testing.assert_allclose(np.asarray(e), np.asarray(j), rtol=1e-12, atol=1e-18)


def test_uneven_sigma_plumbing():
    """``build_dynamics_params`` must thread the sigma-stretching parameters
    into ``compute_vert_coord`` (needed to match Isca's HS ``uneven_sigma``
    config) — not silently fall back to the coordinate defaults."""
    from jsca.dycore.vert_coordinate import compute_vert_coord

    kw = dict(scale_heights=6.0, surf_res=0.5, exponent=7.5)
    pk_ref, bk_ref = compute_vert_coord("uneven_sigma", K, reference_press=1.0e5, **kw)
    p = build_dynamics_params(
        NF, NLAT, NLON, K, vert_coord_option="uneven_sigma",
        reference_sea_level_press=1.0e5, **kw,
    )
    np.testing.assert_allclose(np.asarray(p.pk), pk_ref, rtol=1e-14, atol=0)
    np.testing.assert_allclose(np.asarray(p.bk), bk_ref, rtol=1e-14, atol=0)
    # the stretching parameters must actually change the coordinate (guards
    # against the plumbing being a no-op): differs from the default profile
    _, bk_default = compute_vert_coord("uneven_sigma", K, reference_press=1.0e5)
    assert not np.allclose(bk_ref, bk_default)


def test_rest_state_steady_uneven_sigma():
    """Rest state stays an exact steady state on a stretched coordinate too."""
    p = build_dynamics_params(
        NF, NLAT, NLON, K, vert_coord_option="uneven_sigma",
        scale_heights=6.0, surf_res=0.5, exponent=7.5, reference_sea_level_press=1.0e5,
    )
    wm = build_wave_matrices(p.implicit, DT)
    vor, div, ts, lnps = rest_state(p, NLAT, NLON, temperature=280.0, surface_press=1.0e5)
    vors, divs, tss, lnpss = _stack(vor, div, ts, lnps)
    dvor, ddiv, dts, dlnps = compute_tendencies(p, vors, divs, tss, lnpss, DT, wm, 0, 1)
    for x in (dvor, ddiv, dts, dlnps):
        assert float(jnp.abs(x).max()) < 1e-11
