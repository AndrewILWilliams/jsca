"""jsca-vs-Isca single-column model comparison (Tier-2 validation, issue #43).

Reads a short Isca single-column reference run (daily-mean output from
``run_column_reference.py`` -- the canonical ``column_test.py`` physics) and runs
jsca's SCM (:mod:`jsca.model.column`) with the **same** vertical coordinate,
latitude, timestep and initial condition, then overplots the two spin-up
trajectories and the equilibrium profiles.

The comparison is apples-to-apples by construction: jsca reads Isca's actual
``pk``/``bk`` from the reference file (so it uses whatever coordinate Isca really
ran -- this is also how the ``num_levels`` ambiguity gets resolved), and averages
each field over each model day to match Isca's daily-mean diagnostics.

Usage:
    python scripts/compare_column_scm.py <isca_atmos_daily.nc> [out_png]
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import jsca  # noqa: F401  (enables x64)
from jsca.model import column as C

ROOT = Path(__file__).resolve().parent.parent
isca_nc = Path(sys.argv[1])
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs" / "figures" / "column_scm_vs_isca.png"

# --- Isca reference (single column: lat/lon size 1) ---
ds = xr.open_dataset(isca_nc, decode_times=False)
pk = np.asarray(ds["pk"].values, dtype=float)
bk = np.asarray(ds["bk"].values, dtype=float)
K = bk.size - 1
lat_deg = float(np.asarray(ds["lat"].values).ravel()[0])
n_days = ds.sizes["time"]

isca = {
    "t_surf": np.asarray(ds["t_surf"].values).reshape(n_days, -1)[:, 0],
    "t_air": np.asarray(ds["temp"].values)[:, -1].reshape(n_days, -1)[:, 0],
    "precip": np.asarray(ds["precipitation"].values).reshape(n_days, -1)[:, 0] * 86400.0,
    "T_prof": np.asarray(ds["temp"].values)[-1].reshape(K, -1)[:, 0],
    "q_prof": np.asarray(ds["sphum"].values)[-1].reshape(K, -1)[:, 0] * 1e3,
    "ps": np.asarray(ds["ps"].values).reshape(n_days, -1)[:, 0],
}
p_full_hpa = None
if "pfull" in ds:
    p_full_hpa = np.asarray(ds["pfull"].values, dtype=float)
print(f"Isca: {K} levels, lat={lat_deg:.3f} deg, {n_days} days, "
      f"ps~{isca['ps'][0]:.0f} Pa (pk[0]={pk[0]}, bk[-1]={bk[-1]})")

# --- jsca with the SAME coordinate / lat / dt / IC ---
DT = 1440.0
m = C.build_column(lat_value=lat_deg, dt=DT, pk=pk, bk=bk)
s0 = C.initial_state(m)  # matches column_test IC (T=264, q=1e-3, u_surf=5)
steps_per_day = int(round(86400 / DT))

# cold-start forward step, then daily-mean accumulation to match Isca diagnostics
state = jax.jit(lambda s: C.step(m, s, m.dt))(s0)


def day_means(state, n_days):
    def one_day(s, _):
        def chunk(ss, _):
            s2, precip = C._step_full(m, ss)
            u, v, tg, qg, ps, t_surf = s2
            samp = jnp.array([t_surf.ravel()[0], tg[..., -1, 1].ravel()[0],
                              precip.ravel()[0], ps.ravel()[0]])
            return s2, (samp, tg[..., 1], qg[..., 1])
        s, (samp, T, q) = jax.lax.scan(chunk, s, None, length=steps_per_day)
        return s, (samp.mean(0), T.mean(0), q.mean(0))
    return jax.lax.scan(one_day, state, None, length=n_days)


state, (samp, Tprof, qprof) = jax.jit(lambda s: day_means(s, n_days))(state)
samp = np.asarray(samp)
jsca_d = {"t_surf": samp[:, 0], "t_air": samp[:, 1],
          "precip": samp[:, 2] * 86400.0, "ps": samp[:, 3]}
T_prof_j = np.asarray(Tprof[-1]).reshape(K, -1)[:, 0]
q_prof_j = np.asarray(qprof[-1]).reshape(K, -1)[:, 0] * 1e3

if p_full_hpa is None:
    from jsca.dycore.press_and_geopot import pressure_variables
    _, _, pf, _ = pressure_variables(pk, bk, jnp.asarray(isca["ps"][0]), m.vert_difference_option)
    p_full_hpa = np.asarray(pf).ravel() / 100.0

days = np.arange(1, n_days + 1)


def _rmsd(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


print(f"final-day SST  jsca={jsca_d['t_surf'][-1]:.3f}  isca={isca['t_surf'][-1]:.3f} K")
print(f"final-day precip jsca={jsca_d['precip'][-1]:.3f}  isca={isca['precip'][-1]:.3f} mm/day")
print(f"T-profile RMSD = {_rmsd(T_prof_j, isca['T_prof']):.4f} K   "
      f"q-profile RMSD = {_rmsd(q_prof_j, isca['q_prof']):.4f} g/kg")
print(f"SST(t) RMSD = {_rmsd(jsca_d['t_surf'], isca['t_surf']):.4f} K")

# --- plot ---
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle(f"jsca vs Isca single-column model  ({K} levels, lat={lat_deg:.1f}deg, "
             f"dt={DT:.0f}s, {n_days} days)", fontsize=12)

a = ax[0, 0]
a.plot(days, isca["t_surf"], "o-", ms=3, color="k", label="Isca SST")
a.plot(days, jsca_d["t_surf"], "-", color="tab:red", label="jsca SST")
a.plot(days, isca["t_air"], "o-", ms=3, color="grey", label="Isca air (sfc)")
a.plot(days, jsca_d["t_air"], "-", color="tab:orange", label="jsca air (sfc)")
a.set_xlabel("day"); a.set_ylabel("temperature (K)"); a.set_title("Temperature spin-up")
a.legend(fontsize=8); a.grid(alpha=0.3)

a = ax[0, 1]
a.plot(days, isca["precip"], "o-", ms=3, color="k", label="Isca")
a.plot(days, jsca_d["precip"], "-", color="tab:blue", label="jsca")
a.set_xlabel("day"); a.set_ylabel("precip (mm/day)"); a.set_title("Precipitation")
a.legend(fontsize=8); a.grid(alpha=0.3)

a = ax[1, 0]
a.plot(isca["T_prof"], p_full_hpa, "o-", ms=3, color="k", label="Isca")
a.plot(T_prof_j, p_full_hpa, "-", color="tab:red", label="jsca")
a.invert_yaxis(); a.set_xlabel("temperature (K)"); a.set_ylabel("pressure (hPa)")
a.set_title(f"T profile (day {n_days})"); a.legend(fontsize=8); a.grid(alpha=0.3)

a = ax[1, 1]
a.plot(isca["q_prof"], p_full_hpa, "o-", ms=3, color="k", label="Isca")
a.plot(q_prof_j, p_full_hpa, "-", color="tab:green", label="jsca")
a.invert_yaxis(); a.set_xlabel("specific humidity (g/kg)"); a.set_ylabel("pressure (hPa)")
a.set_title(f"q profile (day {n_days})"); a.legend(fontsize=8); a.grid(alpha=0.3)

fig.tight_layout(rect=(0, 0, 1, 0.95))
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, dpi=120)
print("wrote", out_png)
