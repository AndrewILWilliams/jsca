"""Tier-2 regression test: jsca's single-column model vs a real Isca column run.

Runs jsca's SCM for a 40-day spin-up on Isca's *own* configuration (read from the
committed reference) and asserts it reproduces the Isca daily-mean trajectory and
equilibrium profiles within tolerance. This is the CI gate that keeps the SCM
faithful to Isca as the code evolves.

The reference (``baseline/reference/column_scm_isca_t264.npz``) is distilled from a
real Isca ``column_test.py`` run (pinned commit a290bc3) at the **canonical** column
config. ``scripts/run_isca_column_reference.py`` builds/runs Isca,
``scripts/distill_column_reference.py`` reduces the NetCDF to this numpy file. CI
never needs Isca itself: it validates against the committed golden trajectory
(the same posture as the frierson climatology references). To regenerate the
reference, rebuild Isca and re-run those two scripts.

jsca now runs the **full canonical column config**, closing the jsca-vs-Isca gaps in
order of size: Isca's deliberately-unstable slab init (``t_surf = init_temp + 1 K``,
``idealized_moist_phys.F90`` L643; SST 0.55 K -> 0.02 K), then
``do_lcl_diffusivity_depth`` (boundary-layer depth = convective LCL height, not the
bulk-Richardson PBL) which the golden step fixture localised as the boundary-layer /
tropical residual (``pbl_height`` off by 6.5 m; tightened the tropics ~15x and precip
~100x), plus ``surface_flux use_virtual_temp=True`` and ``lscale_cond do_evap=False``.
Measured agreement now: SST RMSD 0.012 K, T-profile 0.13 K, q 0.11 g/kg, precip
0.005 mm/day -- the tolerances below sit above these.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.model import column as C

REF = Path(__file__).resolve().parent.parent / "baseline" / "reference" / "column_scm_isca_t264.npz"
DT = 1440.0


@pytest.fixture(scope="module")
def isca():
    d = np.load(REF)
    return {k: np.asarray(d[k]) for k in d.files}


@pytest.fixture(scope="module")
def jsca_run(isca):
    """jsca SCM on Isca's coordinate/lat/dt/IC, daily-mean over the run."""
    pk, bk = isca["pk"], isca["bk"]
    lat = float(isca["lat_deg"].ravel()[0])
    n_days = int(isca["n_days"])
    m = C.build_column(lat_value=lat, dt=DT, pk=pk, bk=bk)
    s0 = C.initial_state(m)                    # T=264, q=1e-3, u_surf=5, t_surf=264
    steps_per_day = int(round(86400 / DT))

    state = jax.jit(lambda s: C.step(m, s, m.dt))(s0)   # cold-start forward step

    def one_day(s, _):
        def chunk(ss, _):
            s2, precip = C._step_full(m, ss)
            _, _, tg, qg, _, t_surf = s2
            samp = jnp.array([t_surf.ravel()[0], precip.ravel()[0]])
            return s2, (samp, tg[..., 1], qg[..., 1])
        s, (samp, T, q) = jax.lax.scan(chunk, s, None, length=steps_per_day)
        return s, (samp.mean(0), T.mean(0), q.mean(0))

    _, (samp, Tprof, qprof) = jax.jit(
        lambda s: jax.lax.scan(one_day, s, None, length=n_days))(state)
    samp = np.asarray(samp)
    return {
        "t_surf": samp[:, 0],
        "precip": samp[:, 1],
        "T_prof": np.asarray(Tprof[-1]).reshape(len(bk) - 1, -1)[:, 0],
        "q_prof": np.asarray(qprof[-1]).reshape(len(bk) - 1, -1)[:, 0],
        "m": m,
    }


def _rmsd(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def test_reference_is_isca_31_even_sigma(isca):
    """Guards the resolved ambiguity: Isca ran 31 even-sigma levels."""
    bk = isca["bk"]
    assert bk.size - 1 == 31
    assert np.allclose(isca["pk"], 0.0)
    assert np.allclose(np.diff(bk), 1.0 / 31.0)          # uniform sigma
    assert np.isclose(float(isca["lat_deg"].ravel()[0]),
                      np.rad2deg(np.arcsin(1 / np.sqrt(3))), atol=1e-6)


def test_temperature_profile_matches_isca(isca, jsca_run):
    # measured 0.13 K vs the canonical reference (all namelist options now matched:
    # t_surf init, use_virtual_temp, do_evap, do_lcl_diffusivity_depth).
    assert _rmsd(jsca_run["T_prof"], isca["temp"][-1]) < 0.2      # K


def test_humidity_profile_matches_isca(isca, jsca_run):
    assert _rmsd(jsca_run["q_prof"] * 1e3, isca["sphum"][-1] * 1e3) < 0.2   # g/kg (measured 0.11)


def test_sst_trajectory_matches_isca(isca, jsca_run):
    # SST agreement (measured 0.012 K) after the t_surf and do_lcl_diffusivity_depth fixes
    assert _rmsd(jsca_run["t_surf"], isca["t_surf"]) < 0.05      # K over 40 days
    assert abs(jsca_run["t_surf"][-1] - isca["t_surf"][-1]) < 0.05  # final SST


def test_precip_matches_isca(isca, jsca_run):
    # final-day precip within 0.05 mm/day (measured 0.005); do_lcl_diffusivity_depth
    # closed the boundary-layer PBL-depth gap that drove the precip difference.
    assert abs((jsca_run["precip"][-1] - isca["precip"][-1]) * 86400.0) < 0.05
