"""Tier-1 end-to-end test of the Held-Suarez *time loop* against real-Fortran
step-by-step fixtures.

Unlike the per-routine fixtures, this validates the whole assembled step
(``jsca.model.held_suarez.step``: HS forcing -> dynamical-core tendencies ->
leapfrog/RAW -> mass+energy corrections) against Isca's ``spectral_dynamics``
time loop, one step at a time.

Fixture provenance
------------------
``spectral_dynamics.F90`` is un-stubbable, so the reference is harvested from a
full, instrumented Isca run (the pinned commit ``a290bc3``): the dry Held-Suarez
test case at **T21 L15**, ``uneven_sigma`` (``scale_heights=6``, ``surf_res=0.5``,
``exponent=7.5``), ``damping_order=4``, ``dt=600 s``, run **serial** (1 core, so
the dumped spectral fields are global/undecomposed). ``spectral_dynamics`` was
instrumented to dump the initial spectral state and, after each step, the
prognostic spectral fields ``(vors, divs, ts, ln_ps)``. jsca starts from the
identical initial state and its full step is compared to Isca's after every step.

The driver applies the physics forcing to the **previous** time level (Isca's
``atmosphere.F90`` L304-311) and uses ``delta_t = dt`` on the cold-start step
(``previous == current``) then ``2 dt`` -- this fixture is what pinned both down.

Tolerances: the vorticity, divergence and ``ln ps`` are inversion-limited (the
semi-implicit vertical solve uses LAPACK, not the Fortran Gauss-Jordan; documented
deviation) and accumulate mildly over the 8 steps -- held at 1e-9 / 1e-7 / 1e-5.
Temperature carries a small (~1e-5) residual in the global-mean ``(0,0)`` energy
correction (under investigation), which propagates into the field; held at 5e-4.
"""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.model import build_held_suarez
from jsca.model.held_suarez import step

FIXTURE = Path(__file__).parent / "fixtures" / "spectral_dynamics_step_reference.npz"
pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="step fixtures not generated")

VOR_ATOL = 1e-9
DIV_ATOL = 1e-7
LNPS_ATOL = 1e-5
T_ATOL = 5e-4  # dominated by the (0,0) energy-correction residual


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _build(fx):
    nlat = fx["coriolis"].shape[0]
    nlon = 2 * nlat
    m = build_held_suarez(
        num_fourier=21, nlat=nlat, nlon=nlon, num_levels=int(fx["num_levels"]),
        dt=float(fx["dt_real"]), vert_coord_option="uneven_sigma",
        scale_heights=6.0, surf_res=0.5, exponent=7.5, damping_order=4,
        reference_sea_level_press=1.0e5, robert_coeff=float(fx["robert_coeff"]),
        raw_filter_coeff=float(fx["raw_filter_coeff"]),
    )
    return m


def test_grid_and_coordinate_match_fortran(fx):
    """jsca's Gaussian grid and vertical coordinate reproduce Isca's exactly."""
    m = _build(fx)
    np.testing.assert_allclose(np.asarray(m.dyn.coriolis), fx["coriolis"], atol=1e-14)
    np.testing.assert_allclose(np.asarray(m.dyn.pk), fx["pk"], atol=1e-13)
    np.testing.assert_allclose(np.asarray(m.dyn.bk), fx["bk"], atol=1e-13)


def test_step_by_step_matches_fortran(fx):
    """jsca's assembled HS step reproduces Isca's spectral_dynamics each step."""
    m = _build(fx)
    # Isca's per-step time interval: dt_real on the cold-start step, then 2 dt.
    assert fx["delta_t"][0] == pytest.approx(float(fx["dt_real"]))
    assert fx["prev_eq_cur"][0] == 1.0 and fx["prev_eq_cur"][1] == 0.0

    def stk(a, b):
        return jnp.stack([jnp.asarray(a), jnp.asarray(b)], axis=-1)

    st = (stk(fx["vors_prev"], fx["vors_cur"]), stk(fx["divs_prev"], fx["divs_cur"]),
          stk(fx["ts_prev"], fx["ts_cur"]), stk(fx["lnps_prev"], fx["lnps_cur"]))

    nsteps = fx["vors_out"].shape[0]
    for i in range(nsteps):
        st = step(m, st, m.dt, m.wave_matrix_cold) if i == 0 else step(m, st)
        cur = 1  # after the roll, the new state sits in slot 1
        vor_err = np.abs(np.asarray(st[0][..., cur]) - fx["vors_out"][i]).max()
        div_err = np.abs(np.asarray(st[1][..., cur]) - fx["divs_out"][i]).max()
        t_err = np.abs(np.asarray(st[2][..., cur]) - fx["ts_out"][i]).max()
        lnps_err = np.abs(np.asarray(st[3][..., cur]) - fx["lnps_out"][i]).max()
        assert vor_err < VOR_ATOL, f"step {i + 1} vor {vor_err:.2e}"
        assert div_err < DIV_ATOL, f"step {i + 1} div {div_err:.2e}"
        assert lnps_err < LNPS_ATOL, f"step {i + 1} lnps {lnps_err:.2e}"
        assert t_err < T_ATOL, f"step {i + 1} T {t_err:.2e}"
