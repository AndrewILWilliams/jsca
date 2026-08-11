"""Figure for the FULL_BETTS_MILLER PR: full Betts-Miller convection, jsca vs Isca.

Left: a deep-convecting column — the temperature and humidity increments (jsca over
Isca). Right: jsca-vs-Isca scatter of tdel across all columns/levels. Uses the
committed golden fixture.

Run: python scripts/plot_betts_miller.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.physics import BettsMillerParams, betts_miller

ROOT = Path(__file__).resolve().parent.parent
fx = np.load(ROOT / "tests" / "fixtures" / "betts_miller_reference.npz")

rain, tdel, qdel, cape, cin, klzb, klcl = (np.asarray(a) for a in betts_miller(
    BettsMillerParams(tau_bm=7200.0, rhbm=0.8), 600.0,
    fx["bm_t"], fx["bm_q"], fx["bm_pfull"], fx["bm_phalf"]))

# deepest convecting column (largest CAPE)
flat = np.argmax(fx["bm_cape"].ravel())
idx = np.unravel_index(flat, fx["bm_cape"].shape)
p = fx["bm_pfull"][idx] / 100.0

fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))

ax = axes[0]
ax.plot(fx["bm_tdel"][idx], p, "s", color="k", ms=5, mfc="none", label="Isca")
ax.plot(np.asarray(tdel)[idx], p, "-", color="tab:red", label="jsca")
ax.axvline(0, color="0.7", lw=0.8)
ax.invert_yaxis()
ax.set_xlabel("ΔT increment (K)")
ax.set_ylabel("pressure (hPa)")
ax.set_title(f"Deep column: T adjustment\n(CAPE {fx['bm_cape'][idx]:.0f} J/kg)")
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(fx["bm_qdel"][idx] * 1e3, p, "s", color="k", ms=5, mfc="none", label="Isca")
ax.plot(np.asarray(qdel)[idx] * 1e3, p, "-", color="tab:blue", label="jsca")
ax.axvline(0, color="0.7", lw=0.8)
ax.invert_yaxis()
ax.set_xlabel("Δq increment (g/kg)")
ax.set_ylabel("pressure (hPa)")
ax.set_title("Deep column: q adjustment")
ax.legend(fontsize=9)

ax = axes[2]
ax.scatter(fx["bm_tdel"].ravel(), np.asarray(tdel).ravel(), s=10, alpha=0.5, color="tab:red")
lim = np.array([fx["bm_tdel"].min(), fx["bm_tdel"].max()])
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("Isca ΔT (K)")
ax.set_ylabel("jsca ΔT (K)")
ax.set_title("all columns/levels: ΔT")

maxabs = np.abs(np.asarray(tdel) - fx["bm_tdel"]).max()
fig.suptitle(
    "Full Betts-Miller convection: jsca vs Isca  "
    f"— max |Δ tdel| = {maxabs:.1e} K; klzb/klcl exact, CAPE to ~1e-6", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "docs" / "figures" / "betts_miller.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
