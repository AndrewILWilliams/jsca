"""jsca-vs-Isca single-column latitude-sweep figure (issue #43).

Reads the distilled sweep reference (baseline/reference/column_scm_isca_sweep.npz)
and runs jsca's SCM at each latitude, plotting the equator->pole structure both
models produce: final SST and precip vs latitude, and the day-40 T/q profiles at a
tropical (0 deg) and a high-latitude (60 deg) column.

Run: python scripts/plot_column_sweep.py
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.model import column as C

ROOT = Path(__file__).resolve().parent.parent
d = np.load(ROOT / "baseline" / "reference" / "column_scm_isca_sweep.npz")
pk, bk, lats = d["pk"], d["bk"], d["lat_deg"]
n_days, K = int(d["n_days"]), len(bk) - 1
pfull = d["pfull_hpa"]
DT = 1440.0
steps_per_day = int(round(86400 / DT))


def run(lat):
    m = C.build_column(lat_value=float(lat), dt=DT, pk=pk, bk=bk)
    s0 = C.initial_state(m)
    state = jax.jit(lambda s: C.step(m, s, m.dt))(s0)

    def one_day(s, _):
        def chunk(ss, _):
            s2, precip = C._step_full(m, ss)
            _, _, tg, qg, _, t_surf = s2
            return s2, (jnp.array([t_surf.ravel()[0], precip.ravel()[0]]), tg[..., 1], qg[..., 1])
        s, (samp, T, q) = jax.lax.scan(chunk, s, None, length=steps_per_day)
        return s, (samp.mean(0), T.mean(0), q.mean(0))

    _, (samp, Tp, qp) = jax.jit(lambda s: jax.lax.scan(one_day, s, None, length=n_days))(state)
    samp = np.asarray(samp)
    return (samp[-1, 0], samp[-1, 1], np.asarray(Tp[-1]).reshape(K, -1)[:, 0],
            np.asarray(qp[-1]).reshape(K, -1)[:, 0])


j_sst, j_pr, j_T, j_q = [], [], {}, {}
for i, lat in enumerate(lats):
    sst, pr, T, q = run(lat)
    j_sst.append(sst); j_pr.append(pr * 86400.0)
    j_T[int(lat)] = T; j_q[int(lat)] = q * 1e3

i_sst = d["t_surf"][:, -1]
i_pr = d["precip"][:, -1] * 86400.0

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("jsca vs Isca single-column model — latitude sweep (day 40, 31 levels, dt=1440s)",
             fontsize=12)

a = ax[0, 0]
a.plot(lats, i_sst, "o-", color="k", label="Isca")
a.plot(lats, j_sst, "s--", color="tab:red", label="jsca")
a.set_xlabel("latitude (deg)"); a.set_ylabel("final SST (K)")
a.set_title("Surface temperature vs latitude"); a.legend(); a.grid(alpha=0.3)

a = ax[0, 1]
a.plot(lats, i_pr, "o-", color="k", label="Isca")
a.plot(lats, j_pr, "s--", color="tab:blue", label="jsca")
a.set_xlabel("latitude (deg)"); a.set_ylabel("final precip (mm/day)")
a.set_title("Precipitation vs latitude"); a.legend(); a.grid(alpha=0.3)

a = ax[1, 0]
for lat, col in [(0, "tab:red"), (60, "tab:purple")]:
    ii = list(map(int, lats)).index(lat)
    a.plot(d["temp"][ii][-1], pfull, "o-", ms=3, color="k")
    a.plot(j_T[lat], pfull, "--", color=col, label=f"jsca {lat}°")
a.invert_yaxis(); a.set_xlabel("temperature (K)"); a.set_ylabel("pressure (hPa)")
a.set_title("T profile: tropics vs high-lat (Isca black)"); a.legend(fontsize=8); a.grid(alpha=0.3)

a = ax[1, 1]
for lat, col in [(0, "tab:green"), (60, "tab:olive")]:
    ii = list(map(int, lats)).index(lat)
    a.plot(d["sphum"][ii][-1] * 1e3, pfull, "o-", ms=3, color="k")
    a.plot(j_q[lat], pfull, "--", color=col, label=f"jsca {lat}°")
a.invert_yaxis(); a.set_xlabel("specific humidity (g/kg)"); a.set_ylabel("pressure (hPa)")
a.set_title("q profile: tropics vs high-lat (Isca black)"); a.legend(fontsize=8); a.grid(alpha=0.3)

fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "docs" / "figures" / "column_scm_sweep_vs_isca.png"
fig.savefig(out, dpi=120)
print("wrote", out)
print("SST rmsd vs lat:", float(np.sqrt(np.mean((np.array(j_sst) - i_sst) ** 2))), "K")
