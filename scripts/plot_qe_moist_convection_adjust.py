"""Figure for the qe_moist_convection adjustment (stage 3b) PR: jsca vs Isca.

Left: a deep-convecting column — the temperature and humidity increments the
scheme applies (jsca lines over Isca markers), showing convective heating aloft
and drying. Right: jsca-vs-Isca scatter of deltaT, deltaq and column rain across
all columns.

Run: python scripts/plot_qe_moist_convection_adjust.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.physics.qe_moist_convection import qe_moist_convection

ROOT = Path(__file__).resolve().parent.parent
fx = np.load(ROOT / "tests" / "fixtures" / "qe_moist_convection_reference.npz")

DT = 720.0
rain_j, dT_j, dq_j, cf_j = (np.asarray(a) for a in
                            qe_moist_convection(fx["qe_tin"], fx["qe_qin"],
                                                fx["qe_pfull"], fx["qe_phalf"], DT))
# increments -> rates (K/day, g/kg/day) for readability
dT_fr = fx["qe_deltaT"] / DT * 86400.0
dq_fr = fx["qe_deltaq"] / DT * 86400.0 * 1e3
dT_jr = dT_j / DT * 86400.0
dq_jr = dq_j / DT * 86400.0 * 1e3

# a deep-convecting column (convflag==2) with the most rain
deep = fx["qe_convflag"].astype(int) == 2
rmask = np.where(deep, fx["qe_rain"], -1.0)
i, j = np.unravel_index(np.argmax(rmask), rmask.shape)
p = fx["qe_pfull"][i, j] / 100.0

fig = plt.figure(figsize=(13, 6.5))
gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.1, 1.1], hspace=0.36, wspace=0.36)

axT = fig.add_subplot(gs[:, 0])
axT.plot(dT_fr[i, j], p, "s", color="k", ms=6, mfc="none", label="Isca")
axT.plot(dT_jr[i, j], p, "-", color="tab:orange", label="jsca")
axT.axvline(0, color="0.7", lw=0.8)
axT.invert_yaxis()
axT.set_xlabel("convective heating  dT/dt  (K/day)")
axT.set_ylabel("pressure (hPa)")
axT.annotate(f"Deep-convecting column (i={i}, j={j})\n"
             f"rain = {fx['qe_rain'][i, j] * 86400.0 / DT:.1f} mm/day",
             xy=(0.03, 0.04), xycoords="axes fraction", fontsize=9,
             ha="left", va="bottom")
axT2 = axT.twiny()
axT2.plot(dq_fr[i, j], p, "s", color="k", ms=6, mfc="none")
axT2.plot(dq_jr[i, j], p, "-", color="tab:blue")
axT2.set_xlabel("moistening  dq/dt  (g/kg/day)", color="tab:blue")
axT2.tick_params(axis="x", labelcolor="tab:blue")
axT.plot([], [], "-", color="tab:orange", label="dT/dt (jsca)")
axT.plot([], [], "-", color="tab:blue", label="dq/dt (jsca)")
axT.legend(fontsize=8, loc="lower right")

ax1 = fig.add_subplot(gs[0, 1])
ax1.scatter(dT_fr.ravel(), dT_jr.ravel(), s=6, alpha=0.4, color="tab:orange")
lim = np.array([dT_fr.min(), dT_fr.max()])
ax1.plot(lim, lim, "k-", lw=0.8)
ax1.set_xlabel("Isca dT/dt (K/day)")
ax1.set_ylabel("jsca")
ax1.set_title("all points: heating")

ax2 = fig.add_subplot(gs[1, 1])
ax2.scatter(dq_fr.ravel(), dq_jr.ravel(), s=6, alpha=0.4, color="tab:blue")
lim = np.array([dq_fr.min(), dq_fr.max()])
ax2.plot(lim, lim, "k-", lw=0.8)
ax2.set_xlabel("Isca dq/dt (g/kg/day)")
ax2.set_ylabel("jsca")
ax2.set_title("all points: moistening")

ax3 = fig.add_subplot(gs[:, 2])
rj = rain_j.ravel() * 86400.0 / DT
rf = fx["qe_rain"].ravel() * 86400.0 / DT
ax3.scatter(rf, rj, s=28, alpha=0.7, color="tab:green")
lim = np.array([0, rf.max() * 1.05])
ax3.plot(lim, lim, "k-", lw=0.8)
ax3.set_xlabel("Isca convective rain (mm/day)")
ax3.set_ylabel("jsca convective rain (mm/day)")
ax3.set_title("all columns: rain")

fig.suptitle(
    "Frierson QE convection — Betts-Miller adjustment: jsca vs Isca  "
    f"— max |Δ(dT)| = {np.abs(dT_j - fx['qe_deltaT']).max():.1e} K; convflag exact",
    fontsize=11,
)
fig.subplots_adjust(top=0.88)
out = ROOT / "docs" / "figures" / "qe_moist_convection_adjust.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
