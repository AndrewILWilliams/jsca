"""Figure for the update_tracers PR: grid tracer time-step, jsca vs Isca.

Left: for one column, the humidity at the previous and current levels going in,
and the two outputs — the RAW-filtered current level and the advected future
level — over the column (jsca lines over Isca markers). Right: jsca-vs-Isca
scatter of all three step outputs at every grid point.

Run: python scripts/plot_update_tracers.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.dycore import fv_advection_init, update_grid_tracer

ROOT = Path(__file__).resolve().parent.parent
fx = np.load(ROOT / "tests" / "fixtures" / "update_tracers_reference.npz")


def _t3(a):
    return np.transpose(a, (1, 0, 2))


nx, ny = int(fx["ut_meta"][0]), int(fx["ut_meta"][1])
dt = float(fx["ut_meta"][3])
fv = fv_advection_init(nx, np.asarray(fx["ut_yy"]), degrees_lon=360.0)
q_cur, q_fut, part_filt = update_grid_tracer(
    _t3(fx["ut_q_prev"]), _t3(fx["ut_q_cur_in"]), _t3(fx["ut_dt_tr"]),
    _t3(fx["ut_ug"]), _t3(fx["ut_vg"]), _t3(fx["ut_wg"]), _t3(fx["ut_p_half"]),
    dt, float(fx["ut_robert"][0]), float(fx["ut_raw"][0]), fv, last_step=False)
q_cur, q_fut = np.asarray(q_cur), np.asarray(q_fut)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

i, j = 0, 0
lev = np.arange(int(fx["ut_meta"][2]))
q_prev = _t3(fx["ut_q_prev"])
q_cur_in = _t3(fx["ut_q_cur_in"])
ax = axes[0]
ax.plot(q_prev[i, j] * 1e3, lev, "-", color="0.6", label="q previous (in)")
ax.plot(q_cur_in[i, j] * 1e3, lev, "-", color="tab:purple", label="q current (in)")
ax.plot(_t3(fx["ut_q_cur_out"])[i, j] * 1e3, lev, "s", color="k", ms=6, mfc="none",
        label="Isca out")
ax.plot(q_cur[i, j] * 1e3, lev, "-o", color="tab:blue", ms=4, label="jsca q current (RAW)")
ax.plot(q_fut[i, j] * 1e3, lev, "-^", color="tab:green", ms=4, label="jsca q future (advected)")
ax.invert_yaxis()
ax.set_xlabel("specific humidity (g/kg)")
ax.set_ylabel("model level (0 = top)")
ax.set_title(f"grid tracer step (i={i}, j={j})")
ax.legend(fontsize=7)

ax = axes[1]
for arr, ref, c, lab in [
        (q_cur, "ut_q_cur_out", "tab:blue", "q current (RAW)"),
        (q_fut, "ut_q_fut_out", "tab:green", "q future (advected)"),
        (np.asarray(part_filt), "ut_part_filt", "tab:orange", "RAW increment")]:
    ax.scatter(_t3(fx[ref]).ravel() * 1e3, arr.ravel() * 1e3, s=10, alpha=0.4,
               color=c, label=lab)
allref = _t3(fx["ut_q_fut_out"]).ravel() * 1e3
lim = np.array([allref.min(), allref.max()])
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("Isca output (g/kg)")
ax.set_ylabel("jsca output (g/kg)")
ax.set_title("all grid points")
ax.legend(fontsize=8)

maxabs = max(np.abs(q_cur - _t3(fx["ut_q_cur_out"])).max(),
             np.abs(q_fut - _t3(fx["ut_q_fut_out"])).max())
fig.suptitle(
    "Grid tracer time-step (update_tracers grid branch: advection + Robert/RAW): jsca vs Isca  "
    f"— max |Δ| = {maxabs:.0e}",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "docs" / "figures" / "update_tracers.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
