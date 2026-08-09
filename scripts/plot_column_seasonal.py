"""Figure for the do_seasonal SCM validation: jsca vs Isca over a 90-day seasonal run.

Left: daily-mean TOA insolation marching from NH winter toward summer (jsca over
Isca). Middle: the slab SST trajectory responding to it. Right: the day-90
temperature profile. Uses the committed reference and a live jsca run.

Run: python scripts/plot_column_seasonal.py
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.model import column as C
from jsca.physics.two_stream_gray_rad import _seasonal_insolation

ROOT = Path(__file__).resolve().parent.parent
ref = np.load(ROOT / "baseline" / "reference" / "column_scm_isca_seasonal.npz")
DT = 1440.0
n_days = int(ref["n_days"])
spd = int(round(86400 / DT))

m = C.build_column(lat_value=float(ref["lat_deg"].ravel()[0]),
                   longitude=float(ref["lon_deg"].ravel()[0]),
                   dt=DT, pk=ref["pk"], bk=ref["bk"],
                   do_seasonal=True, solday=-10, equinox_day=0.75, year_in_s=360.0 * 86400.0)

# jsca run: daily-mean SST/precip + day-90 T profile
state = jax.jit(lambda s: C.step(m, s, m.dt, 0.0))(C.initial_state(m))


def one_day(carry, _):
    s, t = carry

    def chunk(cc, _):
        ss, tt = cc
        s2, _p = C._step_full(m, ss, time_seconds=tt)
        _, _, tg, _, _, tsurf = s2
        return (s2, tt + DT), (tsurf.ravel()[0], tg[..., 1])

    (s, t), (ts, T) = jax.lax.scan(chunk, (s, t), None, length=spd)
    return (s, t), (ts.mean(0), T.mean(0))


(_, _), (ts_j, Tprof) = jax.jit(
    lambda st: jax.lax.scan(one_day, (st, m.dt), None, length=n_days))(state)
ts_j = np.asarray(ts_j)
Tprof = np.asarray(Tprof[-1]).reshape(len(ref["bk"]) - 1, -1)[:, 0]

# jsca daily-mean insolation (sampled as Isca time-averages it)
gr, lat2d, lon2d, orb = m.phys.gray_rad, m.lat2d, m.lon2d, m.orb_angle
insol_at = jax.jit(lambda t: _seasonal_insolation(gr, lat2d, lon2d, t, orb, None).ravel()[0])
sw_j = np.array([np.mean([float(insol_at(jnp.asarray((d * spd + k) * DT))) for k in range(spd)])
                 for d in range(n_days)])

days = np.arange(n_days)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

ax = axes[0]
ax.plot(days, ref["swdn_toa"], "s", color="k", ms=4, mfc="none", label="Isca")
ax.plot(days, sw_j, "-", color="tab:orange", label="jsca")
ax.set_xlabel("day")
ax.set_ylabel("TOA insolation (W/m²)")
ax.set_title("Seasonal insolation march\n(NH winter → summer, 35°N)")
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(days, ref["t_surf"], "s", color="k", ms=4, mfc="none", label="Isca")
ax.plot(days, ts_j, "-", color="tab:red", label="jsca")
ax.set_xlabel("day")
ax.set_ylabel("slab SST (K)")
ax.set_title(f"SST response\n(RMSD {np.sqrt(np.mean((ts_j - ref['t_surf'])**2)):.3f} K)")
ax.legend(fontsize=9)

ax = axes[2]
pfull = ref["pfull_hpa"]
ax.plot(ref["temp"][-1], pfull, "s", color="k", ms=4, mfc="none", label="Isca")
ax.plot(Tprof, pfull, "-", color="tab:blue", label="jsca")
ax.invert_yaxis()
ax.set_xlabel("temperature (K)")
ax.set_ylabel("pressure (hPa)")
ax.set_title("Day-90 T profile")
ax.legend(fontsize=9)

fig.suptitle("jsca do_seasonal single-column model vs Isca (90-day seasonal run)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = ROOT / "docs" / "figures" / "column_scm_seasonal_vs_isca.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
