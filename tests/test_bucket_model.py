"""Stability + water-budget smoke test for the land + bucket moist model.

The individual kernels (dynamics, physics, bucket stepping, land mask) are each
golden-fixture-validated elsewhere; this checks that the assembled
:mod:`jsca.model.bucket_model` stepping is self-consistent and stable:

* a short integration stays finite and physical (bucket in ``[0, max]`` over land,
  ocean reservoir effectively infinite/"always wet", humidity non-negative);
* the precipitation → reservoir-depth conversion matches Isca's
  ``depth_change = rain / dens_h2o`` (F90 ``idealized_moist_phys`` L911/L1021):
  ``(depth_change_cond + depth_change_conv) * dens_h2o == precip * delta_t``.

A small T7 / 12-level grid keeps CI fast; the physics fidelity is not re-tested
here (the kernels already are).
"""
import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca import constants
from jsca.grid.gaussian import gaussian_grid
from jsca.model import bucket_model as BM
from jsca.model.idealized_moist_phys import idealized_moist_phys
from jsca.model.land import continents_land_mask


@pytest.fixture(scope="module")
def small_model():
    nlat, nlon = 16, 32
    lat = np.degrees(gaussian_grid(nlat).lat)
    lon = np.arange(nlon) * 360.0 / nlon
    land = continents_land_mask(lat, lon)
    m = BM.build_bucket_model(land, num_fourier=7, nlat=nlat, nlon=nlon, num_levels=12)
    return m, land


def test_short_integration_is_stable_and_physical(small_model):
    m, land = small_model
    state = BM.initial_state(m)
    state = BM.integrate(m, state, 8, cold_start=True)
    *_, t_surf, bucket_depth = state
    qg = state[4]
    bd = np.asarray(bucket_depth[..., 1])
    ts = np.asarray(t_surf)
    q = np.asarray(qg[..., 1])
    landm = land > 0.5

    assert np.isfinite(bd).all() and np.isfinite(ts).all() and np.isfinite(q).all()
    # Reservoir stays within [0, capacity] over land.
    assert bd[landm].min() >= 0.0
    assert bd[landm].max() <= m.base.phys.max_bucket_depth_land + 1e-9
    # Ocean reservoir stays effectively infinite (never dries).
    assert bd[~landm].min() > 100.0
    # Surface temperature and humidity stay physical.
    assert 150.0 < ts.min() and ts.max() < 350.0
    assert q.min() >= 0.0


def test_precip_to_depth_change_conversion(small_model):
    """(cond + conv) reservoir depth added == precip * delta_t / dens_h2o.

    Uses a warm, near-saturated column so convection + large-scale condensation are
    guaranteed to precipitate on this single physics call (the invariant would still
    hold, but vacuously, for a dry column).
    """
    import jax.numpy as jnp

    from jsca.dycore.press_and_geopot import compute_geopotential, pressure_variables
    m, land = small_model
    b = m.base
    dyn = b.dyn
    k = dyn.num_levels
    shape = (b.nlat, b.nlon, k)

    # Warm surface, temperature decreasing with height (so upper levels reach
    # saturation) and a humid boundary layer -> convection + condensation both fire.
    u_p = jnp.zeros(shape)
    v_p = jnp.zeros(shape)
    ps = jnp.full((b.nlat, b.nlon), 1.0e5)
    ph, lph, pf, lpf = pressure_variables(dyn.pk, dyn.bk, ps, dyn.vert_difference_option)
    t_p = jnp.maximum(200.0, 300.0 * (pf / 1.0e5) ** 0.2)
    q_p = jnp.full(shape, 0.018)
    phi_full, phi_half = compute_geopotential(dyn.pk, t_p, lph, lpf, dyn.surf_geopotential)
    t_surf = jnp.full((b.nlat, b.nlon), 301.0)
    gust = b.phys.gust_const * jnp.ones((b.nlat, b.nlon))
    bucket_depth = jnp.full((b.nlat, b.nlon), 1.0)

    out = idealized_moist_phys(
        b.phys, b.lat2d, b.lon2d, u_p, v_p, t_p, q_p, ph, pf, ph, pf,
        phi_full / constants.GRAV, phi_half / constants.GRAV, t_surf, gust,
        b.delta_t, b.dt, land=m.land, bucket_depth=bucket_depth)

    added = np.asarray(out.depth_change_cond + out.depth_change_conv)
    expected = np.asarray(out.precip) * b.delta_t / constants.DENS_H2O
    assert np.allclose(added, expected, rtol=1e-12, atol=1e-18)
    # Non-vacuous: this column really does precipitate.
    assert np.asarray(out.precip).max() > 0.0
