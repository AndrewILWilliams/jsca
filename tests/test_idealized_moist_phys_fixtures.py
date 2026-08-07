"""Tier-1 golden test: the assembled Frierson column physics vs a real Isca step.

This is the end-to-end fidelity gate for the ``idealized_moist_phys`` column driver
(roadmap item 11a): its output is compared against **one real step of the pinned
Isca `frierson_test_case`** at T42, produced by instrumenting the unmodified
``idealized_moist_phys.F90`` to dump the physics I/O (recipe:
``fortran_instrumentation/frierson_step_recipe.md``). Column physics is pointwise,
so the committed fixture subsamples columns (every 4th lat, every 16th lon) to stay
small while spanning all latitudes.

Isca accumulates the physics into ``dt_ug/dt_vg/dt_tg/dt_tracers`` on top of the
values it enters with; the jsca driver returns physics-only tendencies, so the
comparison uses Isca's ``(out - in)``.

Tolerances: momentum matches to machine precision (the sponge is inactive on this
cold-start step, so the wind tendency is pure vertical diffusion). Temperature and
humidity carry the documented ``sat_vapor_pres`` closed-form deviation (~1e-9,
:mod:`jsca.physics.sat_vapor_pres`) through the latent-heat / evaporation terms, so
they are held at the rule-2 deviation tolerance rather than machine epsilon.

Two bugs this fixture caught and fixed (both invisible to the stability smoke
test): the boundary-layer ``diffusivity`` needs the **real geopotential**
half-level heights ``z_half`` (not a midpoint approximation), and ``surface_flux``
needs Isca's initial **gustiness** ``gust = 1.0`` m/s (surface_flux runs before
``vert_turb_driver`` on step 1), both now threaded as inputs.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.model.idealized_moist_phys import FriersonPhysicsParams, idealized_moist_phys
from jsca.physics.damping_driver import damping_driver_init
from jsca.physics.mixed_layer import MixedLayerParams

FIXTURE = Path(__file__).parent / "fixtures" / "idealized_moist_phys_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="idealized_moist_phys Isca fixture not generated"
)


def _t3(a):
    """Fortran (lon, lat, lev) -> jsca (lat, lon, lev)."""
    return np.transpose(a, (1, 0, 2))


def _t2(a):
    return a.T


@pytest.fixture(scope="module")
def result():
    fx = np.load(FIXTURE)
    lat2d, lon2d = _t2(fx["imp_lat"]), _t2(fx["imp_lon"])
    u, v = _t3(fx["imp_u_prev"]), _t3(fx["imp_v_prev"])
    t, q = _t3(fx["imp_t_prev"]), _t3(fx["imp_q_prev"])
    php, pfp = _t3(fx["imp_phalf_prev"]), _t3(fx["imp_pfull_prev"])
    phc, pfc = _t3(fx["imp_phalf_cur"]), _t3(fx["imp_pfull_cur"])
    zsurf = _t2(fx["imp_zsurf"])[..., None]
    zf = _t3(fx["imp_zfull_cur"]) - zsurf
    zh = _t3(fx["imp_zhalf_cur"]) - zsurf
    tsurf = _t2(fx["imp_tsurf_in"])
    gust = np.full(lat2d.shape, 1.0)  # Isca's initial gustiness on step 1
    delta_t = float(fx["imp_delta_t"].ravel()[0])
    dt_real = float(fx["imp_dt_real"].ravel()[0])

    params = FriersonPhysicsParams(
        mixed_layer=MixedLayerParams(depth=2.5, albedo=0.31),
        damping=damping_driver_init(np.asarray(pfc[0, 0])), albedo=0.31)
    out = idealized_moist_phys(
        params, lat2d, lon2d, u, v, t, q, php, pfp, phc, pfc, zf, zh, tsurf, gust,
        delta_t, dt_real)
    return fx, out


def _isca_phys(fx, key):
    """Isca physics-only tendency = accumulated (out - in)."""
    return _t3(fx[key + "_out"]) - _t3(fx[key + "_in"])


@pytest.mark.parametrize("attr,key", [("dt_ug", "imp_dtug"), ("dt_vg", "imp_dtvg")])
def test_momentum_machine_precision(result, attr, key):
    fx, out = result
    jj = np.asarray(getattr(out, attr))
    isca = _isca_phys(fx, key)
    # sponge inactive here -> pure vertical diffusion; machine precision
    np.testing.assert_allclose(jj, isca, rtol=1e-11, atol=1e-16)


@pytest.mark.parametrize("attr,key", [("dt_tg", "imp_dttg"), ("dt_qg", "imp_dtqg")])
def test_thermo_within_documented_deviation(result, attr, key):
    fx, out = result
    jj = np.asarray(getattr(out, attr))
    isca = _isca_phys(fx, key)
    # sat_vapor_pres closed-form deviation (~1e-9) through latent heat / evaporation
    scale = np.abs(isca).max()
    assert np.abs(jj - isca).max() <= 1e-8 * scale + 1e-14


def test_t_surf_matches(result):
    fx, out = result
    np.testing.assert_allclose(
        np.asarray(out.t_surf), _t2(fx["imp_tsurf_out"]), rtol=1e-9, atol=1e-8)
