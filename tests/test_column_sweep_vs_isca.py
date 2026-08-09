"""Tier-2 regression test: jsca's single-column model vs Isca across a latitude sweep.

Extends ``test_column_vs_isca.py`` (single global-average column) to five latitudes
(0, 15, 30, 45, 60 deg), so the physics chain is validated across regimes -- from the
warm/wet tropics (Isca: 299 K, 5.2 mm/day) to the cold/dry high latitudes (267 K,
0.9 mm/day). Each latitude sees a different Frierson p2 insolation, so this catches
regressions that a single column would miss (e.g. a radiation or convection bug that
only bites at low insolation).

Reference: ``baseline/reference/column_scm_isca_sweep.npz``, distilled by
``scripts/distill_column_sweep.py`` from real Isca runs
(``scripts/run_isca_column_sweep.py``). CI stays Isca-free (numpy golden data).

With the full canonical config matched (t_surf init, use_virtual_temp, do_evap, and
do_lcl_diffusivity_depth), agreement is tight at **every** latitude, tropics included:
SST <= 0.018 K, T-profile <= 0.027 K, q <= 0.026 g/kg, precip <= 0.007 mm/day. The
last and largest fix was `do_lcl_diffusivity_depth` (boundary-layer depth = the
convective LCL height, not the bulk-Richardson PBL) -- the golden step fixture
localised the residual to the PBL depth (`pbl_height` off by 6.5 m), and porting it
tightened the tropical profiles ~15x and tropical precip ~100x.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.model import column as C

REF = (Path(__file__).resolve().parent.parent
       / "baseline" / "reference" / "column_scm_isca_sweep.npz")
DT = 1440.0
LAT_INDEX = {0: 0, 15: 1, 30: 2, 45: 3, 60: 4}   # deg -> row in the reference arrays


@pytest.fixture(scope="module")
def ref():
    d = np.load(REF)
    return {k: np.asarray(d[k]) for k in d.files}


def _run_jsca(pk, bk, lat, n_days):
    m = C.build_column(lat_value=float(lat), dt=DT, pk=pk, bk=bk)
    s0 = C.initial_state(m)
    steps_per_day = int(round(86400 / DT))
    state = jax.jit(lambda s: C.step(m, s, m.dt))(s0)

    def one_day(s, _):
        def chunk(ss, _):
            s2, precip = C._step_full(m, ss)
            _, _, tg, qg, _, t_surf = s2
            return s2, (jnp.array([t_surf.ravel()[0], precip.ravel()[0]]),
                        tg[..., 1], qg[..., 1])
        s, (samp, T, q) = jax.lax.scan(chunk, s, None, length=steps_per_day)
        return s, (samp.mean(0), T.mean(0), q.mean(0))

    _, (samp, Tprof, qprof) = jax.jit(
        lambda s: jax.lax.scan(one_day, s, None, length=n_days))(state)
    samp = np.asarray(samp)
    K = len(bk) - 1
    return {
        "t_surf": samp[:, 0], "precip": samp[:, 1],
        "T_prof": np.asarray(Tprof[-1]).reshape(K, -1)[:, 0],
        "q_prof": np.asarray(qprof[-1]).reshape(K, -1)[:, 0],
    }


def _rmsd(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


@pytest.mark.parametrize("lat", [0, 15, 30, 45, 60])
def test_column_matches_isca_at_latitude(ref, lat):
    i = LAT_INDEX[lat]
    n_days = int(ref["n_days"])
    j = _run_jsca(ref["pk"], ref["bk"], ref["lat_deg"][i], n_days)

    # With the full canonical config matched (t_surf init, use_virtual_temp, do_evap,
    # do_lcl_diffusivity_depth), agreement is tight at *every* latitude, tropics
    # included: measured SST <= 0.018 K, T-profile <= 0.027 K, q <= 0.026 g/kg,
    # precip <= 0.007 mm/day. do_lcl_diffusivity_depth (PBL depth = convective LCL
    # height) was the boundary-layer / tropical residual, tightening the tropics ~15x.
    assert _rmsd(j["t_surf"], ref["t_surf"][i]) < 0.05                     # K, trajectory
    assert abs(j["t_surf"][-1] - ref["t_surf"][i][-1]) < 0.05            # K, final SST
    assert _rmsd(j["T_prof"], ref["temp"][i][-1]) < 0.1                    # K, profile
    assert _rmsd(j["q_prof"] * 1e3, ref["sphum"][i][-1] * 1e3) < 0.1      # g/kg, profile
    assert abs((j["precip"][-1] - ref["precip"][i][-1]) * 86400.0) < 0.05  # mm/day


def test_sweep_reproduces_isca_meridional_gradient(ref):
    """jsca reproduces the equator->pole cooling/drying gradient Isca shows
    (a coarse check that the across-regime *structure*, not just each point, is right)."""
    n_days = int(ref["n_days"])
    sst = [_run_jsca(ref["pk"], ref["bk"], ref["lat_deg"][i], n_days)["t_surf"][-1]
           for i in range(len(ref["lat_deg"]))]
    # monotonically decreasing from equator to 60 deg, in both models
    assert np.all(np.diff(sst) < 0)
    assert np.all(np.diff(ref["t_surf"][:, -1]) < 0)
