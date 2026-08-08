"""Smoke tests for the composable object API (:mod:`jsca.api`, :mod:`jsca.configs`).

These guard the wiring: (a) component configs thread through to the frozen params
pytree the functional core builds, and (b) the assembler drives the same stable
integration as the functional ``build_*``/``integrate`` path. Numerical fidelity
is owned by the kernel fixture tests and ``tests/test_{frierson,held_suarez}.py``.
"""
import numpy as np
import pytest

import jsca


def test_moist_config_threads_to_params():
    physics = jsca.MoistPhysics(
        radiation=jsca.GrayRadiation(solar_constant=1400.0, ir_tau_eq=5.0),
        surface=jsca.SurfaceMixedLayer(depth=10.0, albedo=0.25,
                                       monin_obukhov=jsca.api.MOParams(rich_crit=1.5)),
        sponge=jsca.RayleighSponge(trayfric=-0.5),
    )
    sim = jsca.Model(jsca.SpectralGrid(trunc=10, dt=720.0), physics=physics).initialize()
    phys = sim._params.phys
    assert phys.gray_rad.solar_constant == 1400.0
    assert phys.gray_rad.ir_tau_eq == 5.0
    assert phys.mixed_layer.depth == 10.0
    assert phys.mixed_layer.albedo == 0.25
    assert phys.albedo == 0.25          # kept consistent with the surface albedo
    assert phys.mo.rich_crit == 1.5
    assert phys.damping is not None      # sponge built from the reference profile
    # trayfric=-0.5 day -> rfactr = (1/0.5)/86400 s^-1
    assert np.isclose(phys.damping.rfactr, (1.0 / 0.5) / 86400.0)


def test_dry_forcing_threads_to_params():
    physics = jsca.DryForcing(forcing=jsca.HeldSuarezForcing(delh=45.0))
    sim = jsca.Model(jsca.SpectralGrid(trunc=8, dt=1200.0, nlev=8), physics=physics).initialize()
    assert sim._params.hs.delh == 45.0
    assert sim.model.physics.has_moisture is False


def test_frierson_recipe_matches_functional_core():
    """jsca.configs.frierson() reproduces the functional build_frierson path."""
    from jsca.model.frierson import build_frierson, initial_state, integrate

    sim = jsca.configs.frierson(trunc=10, dt=720.0).initialize(humidity=1.0e-3)
    sim.run(steps=6)

    m = build_frierson(num_fourier=10, dt=720.0)  # damping_order=4 by default
    s = integrate(m, initial_state(m, humidity=1.0e-3), n_steps=6, cold_start=True)

    assert sim.n_steps == 6
    _, _, t_ref, _ = _grid_fri(m, s)
    np.testing.assert_array_equal(sim.state.temperature, np.asarray(t_ref))


def test_held_suarez_recipe_runs_and_matches_core():
    from jsca.model.held_suarez import build_held_suarez, initial_state, integrate

    sim = jsca.configs.held_suarez(trunc=8, dt=1200.0, num_levels=8).initialize()
    sim.run(steps=5)

    m = build_held_suarez(num_fourier=8, num_levels=8, dt=1200.0, damping_order=2)
    s = integrate(m, initial_state(m), n_steps=5, cold_start=True)

    _, _, t_ref, _ = _grid_hs(m, s)
    np.testing.assert_array_equal(sim.state.temperature, np.asarray(t_ref))


def test_state_accessors_and_moisture_guard():
    sim = jsca.configs.frierson(trunc=8, dt=1200.0).initialize()
    sim.run(steps=4)
    st = sim.state
    nlat, nlon, k = st.temperature.shape
    assert (nlat, nlon) == (18, 36)     # 2*8+2, 4*8+4
    for f in (st.u, st.v, st.temperature, st.sphum):
        assert f.shape == (nlat, nlon, k)
        assert np.all(np.isfinite(f))
    assert st.t_surf.shape == (nlat, nlon)

    dry = jsca.configs.held_suarez(trunc=8, dt=1200.0, num_levels=8).initialize()
    dry.run(steps=3)
    with pytest.raises(AttributeError, match="moist"):
        _ = dry.state.sphum


def test_climatology_returns_fields():
    sim = jsca.configs.frierson(trunc=8, dt=1200.0).initialize()
    clim = sim.climatology(spinup_days=0.1, avg_days=0.1)
    for key in ("ucomp", "vcomp", "temp", "sphum", "ps", "t_surf", "precip"):
        assert key in clim and np.all(np.isfinite(clim[key]))


def test_moist_nlev_guard():
    with pytest.raises(ValueError, match="pinned"):
        jsca.Model(jsca.SpectralGrid(trunc=8, nlev=20), physics=jsca.MoistPhysics()).initialize()


def _grid_fri(m, s):
    from jsca.model.frierson import _grid_from_spectral
    return _grid_from_spectral(m, s[0], s[1], s[2], s[3], 1)


def _grid_hs(m, s):
    from jsca.model.held_suarez import _grid_from_spectral
    return _grid_from_spectral(m, s[0], s[1], s[2], s[3], 1)
