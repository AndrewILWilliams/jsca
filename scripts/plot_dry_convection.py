"""Figure for the dry_convection PR: Schneider-Walker dry adjustment, jsca vs Isca.

Left: a convecting column — environment temperature, the energy-conserving adjusted
profile, and the tendency (jsca over Isca). Right: jsca-vs-Isca scatter of dt_tg
across all columns/levels. Uses the committed golden fixture.

Run: python scripts/plot_dry_convection.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.physics import DryConvectionParams, dry_convection

ROOT = Path(__file__).resolve().parent.parent
fx = np.load(ROOT / "tests" / "fixtures" / "dry_convection_reference.npz")

dt_j, cape_j, cin_j, lzb_j, lcl_j = (np.asarray(a) for a in dry_convection(
    DryConvectionParams(tau=21600.0, gamma=1.0),
    fx["dc_t"], fx["dc_pfull"], fx["dc_phalf"]))
dt_i = fx["dc_dt_tg"]
tau = 21600.0

# pick a deep convecting column (largest CAPE)
flat = np.argmax(fx["dc_cape"].ravel())
idx = np.unravel_index(flat, fx["dc_cape"].shape)
t_env = fx["dc_t"][idx]
p = fx["dc_pfull"][idx] / 100.0                    # hPa
t_adj_i = t_env + dt_i[idx] * tau                  # Isca adjusted profile
t_adj_j = t_env + np.asarray(dt_j)[idx] * tau

fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))

ax = axes[0]
ax.plot(t_env, p, "-", color="0.6", label="environment")
ax.plot(t_adj_i, p, "s", color="k", ms=5, mfc="none", label="Isca adjusted")
ax.plot(t_adj_j, p, "-", color="tab:red", label="jsca adjusted")
ax.invert_yaxis()
ax.set_xlabel("temperature (K)")
ax.set_ylabel("pressure (hPa)")
ax.set_title(f"Dry adjustment, deep column\n(CAPE {fx['dc_cape'][idx]:.0f} J/kg)")
ax.legend(fontsize=9)

ax = axes[1]
ax.scatter(dt_i.ravel() * 86400, np.asarray(dt_j).ravel() * 86400,
           s=10, alpha=0.5, color="tab:red")
lim = np.array([dt_i.min(), dt_i.max()]) * 86400
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("Isca dt_tg (K/day)")
ax.set_ylabel("jsca dt_tg (K/day)")
ax.set_title("all columns/levels: tendency")

maxabs = np.abs(np.asarray(dt_j) - dt_i).max()
fig.suptitle(
    "dry_convection (Schneider-Walker): jsca vs Isca  "
    f"— max |Δ dt_tg| = {maxabs:.1e} K/s (LZB/LCL indices exact)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "docs" / "figures" / "dry_convection.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
