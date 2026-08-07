"""Stability / conservation smoke test for the Frierson column-physics driver.

The individual modules composed by :func:`idealized_moist_phys` are each golden-
fixture-validated against the real Fortran; this test gates the *assembly* (call
order + tendency bookkeeping) on a synthetic but physical global column set:

* no NaN/Inf anywhere in the returned tendencies or the updated SST;
* precipitation is non-negative;
* the SST increment over one step is bounded (slab-ocean heat capacity);
* the tendencies have physically sane magnitudes (no runaway);
* the driver is jit-safe.

An end-to-end *golden* step fixture from an instrumented Isca Frierson run is the
remaining validation (roadmap item 11); it needs a full Isca build.
"""
import jax
import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.dycore.press_and_geopot import pressure_variables
from jsca.model.idealized_moist_phys import FriersonPhysicsParams, idealized_moist_phys
from jsca.physics.damping_driver import damping_driver_init

# Frierson pure-sigma 25-level coordinate (frierson_test_case.py: pk = 0, bk given)
_BK = np.array([
    0.000000, 0.0117665, 0.0196679, 0.0315244, 0.0485411, 0.0719344, 0.1027829,
    0.1418581, 0.1894648, 0.2453219, 0.3085103, 0.3775033, 0.4502789, 0.5244989,
    0.5977253, 0.6676441, 0.7322627, 0.7900587, 0.8400683, 0.8819111, 0.9157609,
    0.9422770, 0.9625127, 0.9778177, 0.9897489, 1.0000000])
_PK = np.zeros(26)

NLAT, NLON, K = 8, 16, 25
DT = 720.0
DELTA_T = 2.0 * DT


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    lat = np.linspace(-80, 80, NLAT) * np.pi / 180.0
    lat2d = np.broadcast_to(lat[:, None], (NLAT, NLON))
    lon2d = np.broadcast_to(np.linspace(0, 2 * np.pi, NLON)[None, :], (NLAT, NLON))

    ps = 1.0e5 * np.ones((NLAT, NLON))
    p_half, _, p_full, _ = pressure_variables(_PK, _BK, ps, "simmons_and_burridge")

    sigma = p_full / ps[..., None]
    # a warm, moist, statically-stable-ish tropospheric column
    t = 200.0 + 90.0 * sigma
    # humidity decreasing with height, a few g/kg near the surface
    q = np.clip(0.015 * sigma**3, 1e-8, None)
    u = 5.0 + 20.0 * sigma + 0.5 * rng.standard_normal((NLAT, NLON, K))
    v = 2.0 * rng.standard_normal((NLAT, NLON, K))
    # geopotential heights above surface from hydrostatic ~ -H ln(sigma)
    sigma_h = p_half / ps[..., None]
    z_full = -8000.0 * np.log(sigma)
    z_half = -8000.0 * np.log(np.clip(sigma_h, 1e-6, None))
    t_surf = 285.0 + 15.0 * np.cos(lat)[:, None] * np.ones((1, NLON))  # warm tropics
    gust = np.full((NLAT, NLON), 1.0)

    pref = np.asarray(p_full[0, 0])  # a reference column for the sponge depth
    params = FriersonPhysicsParams(damping=damping_driver_init(pref))
    return dict(params=params, lat2d=lat2d, lon2d=lon2d, u=u, v=v, t=t, q=q,
                p_half=p_half, p_full=p_full, z_full=z_full, z_half=z_half,
                t_surf=t_surf, gust=gust)


def _run(s):
    return idealized_moist_phys(
        s["params"], s["lat2d"], s["lon2d"], s["u"], s["v"], s["t"], s["q"],
        s["p_half"], s["p_full"], s["p_half"], s["p_full"], s["z_full"], s["z_half"],
        s["t_surf"], s["gust"], DELTA_T, DT)


def test_no_nans(setup):
    out = _run(setup)
    for name in ("dt_ug", "dt_vg", "dt_tg", "dt_qg", "t_surf", "precip", "pbl_height"):
        a = np.asarray(getattr(out, name))
        assert np.all(np.isfinite(a)), f"{name} has non-finite values"


def test_precip_nonnegative(setup):
    out = _run(setup)
    assert np.all(np.asarray(out.precip) >= 0.0)


def test_sst_increment_bounded(setup):
    """One 720 s step moves SST by well under a degree (slab-ocean heat capacity)."""
    out = _run(setup)
    dts = np.asarray(out.t_surf) - setup["t_surf"]
    assert np.all(np.abs(dts) < 1.0)


def test_tendency_magnitudes_sane(setup):
    """Tendencies stay bounded (runaway catcher). These are loose bounds: a
    cold-start step on a synthetic, convectively-active column legitimately
    produces large adjustment tendencies; the point is that nothing diverges. The
    tight, exact check is the golden step fixture (roadmap item 11)."""
    out = _run(setup)
    assert np.abs(np.asarray(out.dt_tg)).max() * 86400.0 < 2000.0   # K/day
    assert np.abs(np.asarray(out.dt_ug)).max() * 86400.0 < 2000.0   # m/s/day
    assert np.abs(np.asarray(out.dt_qg)).max() * 86400.0 < 0.5      # kg/kg/day


def test_jit(setup):
    s = setup
    f = jax.jit(lambda u, v, t, q, ts: idealized_moist_phys(
        s["params"], s["lat2d"], s["lon2d"], u, v, t, q,
        s["p_half"], s["p_full"], s["p_half"], s["p_full"], s["z_full"], s["z_half"],
        ts, s["gust"], DELTA_T, DT))
    out = f(s["u"], s["v"], s["t"], s["q"], s["t_surf"])
    assert np.all(np.isfinite(np.asarray(out.dt_tg)))
    # jit result must match eager (no tracing surprises)
    eager = _run(s)
    np.testing.assert_allclose(np.asarray(out.dt_tg), np.asarray(eager.dt_tg), rtol=1e-12)
