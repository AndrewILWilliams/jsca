"""Smoke tests for the prototype object API (:mod:`jsca.api`).

These check that the ergonomics layer (a) threads notebook-set config through to
the frozen params pytree the functional core builds, and (b) drives the same
stable integration as the functional `build_frierson`/`integrate` path. Numerical
fidelity is owned by the kernel fixture tests and `tests/test_frierson.py`; this
file only guards the wiring.
"""
import numpy as np

import jsca
from jsca.model.frierson import build_frierson, initial_state, integrate


def test_config_threads_to_params():
    """Component overrides land on the built FriersonPhysicsParams."""
    model = jsca.Frierson(
        jsca.SpectralGrid(trunc=10, dt=720.0),
        radiation=jsca.GrayRadiation(solar_constant=1400.0, ir_tau_eq=5.0),
        ocean=jsca.MixedLayer(depth=10.0, albedo=0.25),
        surface_layer=jsca.SurfaceLayer(rich_crit=1.5),
    )
    sim = model.initialize()
    phys = sim._params.phys
    assert phys.gray_rad.solar_constant == 1400.0
    assert phys.gray_rad.ir_tau_eq == 5.0
    assert phys.mixed_layer.depth == 10.0
    assert phys.mixed_layer.albedo == 0.25
    assert phys.albedo == 0.25            # kept consistent with the ocean albedo
    assert phys.mo.rich_crit == 1.5
    assert phys.damping is not None        # sponge filled from the reference profile


def test_run_matches_functional_core():
    """The object API produces the identical state to the functional path."""
    grid = jsca.SpectralGrid(trunc=10, dt=720.0)
    sim = jsca.Frierson(grid).initialize(humidity=1.0e-3)
    sim.run(steps=6)

    m = build_frierson(num_fourier=10, dt=720.0)
    s = integrate(m, initial_state(m, humidity=1.0e-3), n_steps=6, cold_start=True)

    assert sim.n_steps == 6
    assert np.isclose(sim.day, 6 * 720.0 / 86400.0)
    # tensordot-free field check: temperatures agree bit-for-bit with the core path
    _, _, t_ref, _ = _grid(m, s)
    np.testing.assert_array_equal(sim.state.temperature, np.asarray(t_ref))


def test_state_accessors_shapes_and_finite():
    sim = jsca.Frierson(jsca.SpectralGrid(trunc=8, dt=1200.0)).initialize()
    sim.run(steps=4)
    st = sim.state
    nlat, nlon, k = st.temperature.shape
    assert (nlat, nlon) == (18, 36)        # 2*8+2, 4*8+4
    for f in (st.u, st.v, st.temperature, st.sphum):
        assert f.shape == (nlat, nlon, k)
        assert np.all(np.isfinite(f))
    assert st.surface_pressure.shape == (nlat, nlon)
    assert st.t_surf.shape == (nlat, nlon)


def test_climatology_returns_fields():
    sim = jsca.Frierson(jsca.SpectralGrid(trunc=8, dt=1200.0)).initialize()
    clim = sim.climatology(spinup_days=0.1, avg_days=0.1)
    for key in ("ucomp", "vcomp", "temp", "sphum", "ps", "t_surf", "precip"):
        assert key in clim
        assert np.all(np.isfinite(clim[key]))


def test_nlev_guard():
    import pytest
    with pytest.raises(ValueError, match="pinned"):
        jsca.Frierson(jsca.SpectralGrid(trunc=8, nlev=20)).initialize()


def _grid(m, s):
    from jsca.model.frierson import _grid_from_spectral
    return _grid_from_spectral(m, s[0], s[1], s[2], s[3], 1)
