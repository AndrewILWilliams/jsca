"""Figure for the lscale_cond PR: jsca vs Isca large-scale condensation.

Left: a representative column — input humidity vs saturation, and the resulting
temperature/humidity increments (jsca lines over Isca markers), showing
condensation removing supersaturation and re-evaporation moistening a dry layer.
Right: jsca-vs-Isca scatter across all points for qdel, tdel, and column rain.

Run: python scripts/plot_lscale_cond.py  (writes docs/figures/lscale_cond.png)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.physics import lscale_cond
from jsca.physics.sat_vapor_pres import saturation_specific_humidity

ROOT = Path(__file__).resolve().parent.parent
fx = np.load(ROOT / "tests" / "fixtures" / "lscale_cond_reference.npz")

tin, qin = fx["lc_tin"], fx["lc_qin"]
pfull, phalf = fx["lc_pfull"], fx["lc_phalf"]
rain_f, tdel_f, qdel_f = fx["lc_rain"], fx["lc_tdel"], fx["lc_qdel"]

rain_j, tdel_j, qdel_j = (np.asarray(a) for a in
                          lscale_cond(tin, qin, pfull, phalf, hc=1.0, do_evap=True))

# saturation humidity for the profile panel (jsca closed form)
qsat = np.asarray(saturation_specific_humidity(tin, pfull))

# pick the column with the most re-evaporation activity to show all branches
reevap_per_col = (qdel_f > 0).sum(axis=-1)
i, j = np.unravel_index(np.argmax(reevap_per_col), reevap_per_col.shape)
p = pfull[i, j] / 100.0  # hPa

fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.1, 1.2], hspace=0.32, wspace=0.34)

# --- column: humidity vs saturation ---
ax0 = fig.add_subplot(gs[:, 0])
ax0.plot(qin[i, j] * 1e3, p, "o-", color="tab:blue", label="q$_{in}$", ms=4)
ax0.plot(qsat[i, j] * 1e3, p, "-", color="tab:red", label="q$_{sat}$")
ax0.fill_betweenx(p, qsat[i, j] * 1e3, qin[i, j] * 1e3,
                  where=qin[i, j] > qsat[i, j], color="tab:red", alpha=0.15,
                  label="supersaturated")
ax0.invert_yaxis()
ax0.set_xlabel("specific humidity (g/kg)")
ax0.set_ylabel("pressure (hPa)")
ax0.set_title(f"Column (i={i}, j={j}): input state")
ax0.legend(fontsize=8)

# --- column: increments jsca vs Isca ---
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(qdel_f[i, j] * 1e3, p, "s", color="k", ms=6, mfc="none", label="Isca")
ax1.plot(qdel_j[i, j] * 1e3, p, "-", color="tab:blue", label="jsca")
ax1.axvline(0, color="0.7", lw=0.8)
ax1.invert_yaxis()
ax1.set_xlabel("Δq (g/kg)")
ax1.set_ylabel("pressure (hPa)")
ax1.set_title("humidity increment")
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(tdel_f[i, j], p, "s", color="k", ms=6, mfc="none", label="Isca")
ax2.plot(tdel_j[i, j], p, "-", color="tab:orange", label="jsca")
ax2.axvline(0, color="0.7", lw=0.8)
ax2.invert_yaxis()
ax2.set_xlabel("ΔT (K)")
ax2.set_ylabel("pressure (hPa)")
ax2.set_title("temperature increment")
ax2.legend(fontsize=8)

# --- scatter: jsca vs Isca across all points ---
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(qdel_f.ravel() * 1e3, qdel_j.ravel() * 1e3, s=6, alpha=0.4,
            color="tab:blue")
lim = np.array([qdel_f.min(), qdel_f.max()]) * 1e3
ax3.plot(lim, lim, "k-", lw=0.8)
ax3.set_xlabel("Isca Δq (g/kg)")
ax3.set_ylabel("jsca Δq (g/kg)")
ax3.set_title("all points: Δq")

ax4 = fig.add_subplot(gs[1, 2])
ax4.scatter(rain_f.ravel(), rain_j.ravel(), s=14, alpha=0.6, color="tab:green")
rlim = np.array([0, rain_f.max() * 1.05])
ax4.plot(rlim, rlim, "k-", lw=0.8)
ax4.set_xlabel("Isca rain (kg/m$^2$)")
ax4.set_ylabel("jsca rain (kg/m$^2$)")
ax4.set_title("all columns: rain")

# relative diff over cells with a meaningful increment (avoid dividing near-zero)
sig = np.abs(qdel_f) > 1e-6
maxrel = np.max(np.abs(qdel_j - qdel_f)[sig] / np.abs(qdel_f)[sig])
fig.suptitle(
    "Large-scale condensation (lscale_cond, do_simple/do_evap): jsca vs Isca  "
    f"— max rel. diff in Δq = {maxrel:.1e} (sat_vapor_pres table deviation)",
    fontsize=11,
)

out = ROOT / "docs" / "figures" / "lscale_cond.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
