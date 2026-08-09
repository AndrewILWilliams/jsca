"""Tier-2 regression test: jsca's single-column model with ``do_seasonal`` vs Isca.

Validates the seasonal + diurnal insolation path end-to-end. jsca runs its SCM for
90 days on Isca's own ``do_seasonal=.true.`` column configuration and must
reproduce Isca's daily-mean trajectory as the season marches on. This is the CI
gate that keeps the seasonal insolation (astronomy ``diurnal_solar`` + the
``two_stream_gray_rad`` seasonal branch + the column-model time threading) faithful.

Reference: ``baseline/reference/column_scm_isca_seasonal.npz``, distilled by
``scripts/distill_column_seasonal.py`` from a real Isca column run
(``scripts/run_isca_column_seasonal.py``; pinned commit a290bc3, ``thirty_day``
calendar, ``current_date=[1,1,1]``). The run starts in NH winter at
lat = arcsin(1/sqrt3) ~ 35.3 deg N; over 90 days the daily-mean TOA insolation
climbs from ~180 to ~358 W/m^2 and the slab SST warms from ~265 to ~284 K.

Measured agreement (jsca vs Isca over the 90 days):
  * TOA insolation (swdn_toa): interior days match to ~machine precision; the two
    run-boundary days differ by a few W/m^2 purely from the daily-mean averaging
    window at the ends (Isca's last output includes the final end-of-run step that
    jsca's per-day window omits, etc.) -- a diagnostic convention, not physics.
  * SST trajectory RMSD 0.008 K (max 0.016 K); precip within 0.05 mm/day;
    day-90 profiles T 0.07 K, q 0.06 g/kg.
The tolerances below sit above these.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.model import column as C
from jsca.physics.two_stream_gray_rad import _seasonal_insolation

REF = (Path(__file__).resolve().parent.parent
       / "baseline" / "reference" / "column_scm_isca_seasonal.npz")
DT = 1440.0
YEAR_IN_S = 360.0 * 86400.0    # thirty_day calendar


@pytest.fixture(scope="module")
def isca():
    d = np.load(REF)
    return {k: np.asarray(d[k]) for k in d.files}


@pytest.fixture(scope="module")
def model(isca):
    return C.build_column(
        lat_value=float(isca["lat_deg"].ravel()[0]),
        longitude=float(isca["lon_deg"].ravel()[0]),
        dt=DT, pk=isca["pk"], bk=isca["bk"],
        do_seasonal=True, solday=-10, equinox_day=0.75, year_in_s=YEAR_IN_S)


@pytest.fixture(scope="module")
def jsca_run(isca, model):
    """jsca seasonal SCM: daily-mean t_surf/precip and the day-90 T/q profiles."""
    m = model
    n_days = int(isca["n_days"])
    steps_per_day = int(round(86400 / DT))
    s0 = C.initial_state(m)
    state = jax.jit(lambda s: C.step(m, s, m.dt, 0.0))(s0)   # cold-start forward step

    def one_day(carry, _):
        s, t = carry

        def chunk(cc, _):
            ss, tt = cc
            s2, precip = C._step_full(m, ss, time_seconds=tt)
            _, _, tg, qg, _, t_surf = s2
            samp = jnp.array([t_surf.ravel()[0], precip.ravel()[0]])
            return (s2, tt + DT), (samp, tg[..., 1], qg[..., 1])

        (s, t), (samp, T, q) = jax.lax.scan(chunk, (s, t), None, length=steps_per_day)
        return (s, t), (samp.mean(0), T.mean(0), q.mean(0))

    (_, _), (samp, Tprof, qprof) = jax.jit(
        lambda st: jax.lax.scan(one_day, (st, m.dt), None, length=n_days))(state)
    samp = np.asarray(samp)
    K = len(isca["bk"]) - 1
    return {
        "t_surf": samp[:, 0],
        "precip": samp[:, 1],
        "T_prof": np.asarray(Tprof[-1]).reshape(K, -1)[:, 0],
        "q_prof": np.asarray(qprof[-1]).reshape(K, -1)[:, 0],
    }


@pytest.fixture(scope="module")
def jsca_insol(isca, model):
    """jsca daily-mean TOA insolation, sampled the same way Isca time-averages it."""
    m = model
    gr, lat2d, lon2d, orb = m.phys.gray_rad, m.lat2d, m.lon2d, m.orb_angle
    spd = int(round(86400 / DT))
    n_days = int(isca["n_days"])
    insol_at = jax.jit(lambda t: _seasonal_insolation(gr, lat2d, lon2d, t, orb, None).ravel()[0])
    out = np.zeros(n_days)
    for day in range(n_days):
        out[day] = np.mean([float(insol_at(jnp.asarray((day * spd + k) * DT)))
                            for k in range(spd)])
    return out


def _rmsd(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def test_reference_shows_a_seasonal_cycle(isca):
    """Guard: the reference genuinely varies through the season (else nothing is
    being tested). Winter->summer, insolation and SST both climb substantially."""
    sw = isca["swdn_toa"]
    assert sw[-1] - sw[0] > 100.0                       # >100 W/m^2 seasonal swing
    assert isca["t_surf"][-1] - isca["t_surf"][0] > 10.0  # >10 K seasonal warming


def test_seasonal_insolation_matches_isca(isca, jsca_insol):
    """The seasonal march of daily-mean TOA insolation matches Isca. Interior days
    match to ~machine precision; the two boundary days carry only the daily-mean
    averaging-window convention (documented in the module docstring)."""
    sw_isca = isca["swdn_toa"]
    # interior: tight (the astronomy + time threading is exact here)
    interior = slice(1, len(sw_isca) - 1)
    assert np.max(np.abs(jsca_insol[interior] - sw_isca[interior])) < 0.1   # W/m^2
    # whole trajectory including the two endpoints
    assert _rmsd(jsca_insol, sw_isca) < 1.0                                 # W/m^2


def test_sst_trajectory_matches_isca(isca, jsca_run):
    # the SST follows the seasonal insolation; measured RMSD 0.008 K, max 0.016 K
    assert _rmsd(jsca_run["t_surf"], isca["t_surf"]) < 0.05
    assert abs(jsca_run["t_surf"][-1] - isca["t_surf"][-1]) < 0.05


def test_precip_trajectory_matches_isca(isca, jsca_run):
    # daily-mean precip over the whole run (measured max diff 0.042 mm/day)
    assert np.max(np.abs((jsca_run["precip"] - isca["precip"]) * 86400.0)) < 0.1


def test_final_profiles_match_isca(isca, jsca_run):
    assert _rmsd(jsca_run["T_prof"], isca["temp"][-1]) < 0.2                # K
    assert _rmsd(jsca_run["q_prof"] * 1e3, isca["sphum"][-1] * 1e3) < 0.2   # g/kg
