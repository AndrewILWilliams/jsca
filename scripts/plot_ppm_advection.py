"""Figure for the PPM vertical-advection PR: finite-volume parabolic, jsca vs Isca.

Left: the vertical advective tendency profile for one strong-wind column (Courant
> 1, so the multi-cell departure-point integral is in play) — jsca over Isca.
Right: jsca-vs-Isca scatter of the tendency at every grid point, across both the
gentle (Courant < 1) and strong (Courant > 1) regimes.

Run: python scripts/plot_ppm_advection.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.dycore.vert_advection import (
    ADVECTIVE_FORM,
    FINITE_VOLUME_PARABOLIC,
    vert_advection,
)

ROOT = Path(__file__).resolve().parent.parent
fx = np.load(ROOT / "tests" / "fixtures" / "ppm_advection_reference.npz")

DT = 600.0
w, dz, r = fx["ppm_w"], fx["ppm_dz"], fx["ppm_r"]
adv = np.asarray(vert_advection(DT, w, dz, r, scheme=FINITE_VOLUME_PARABOLIC, form=ADVECTIVE_FORM))
k = r.shape[-1]

# a strong-wind column (second half) with the largest Courant number
cn_col = (DT * np.maximum(w[..., 1:k], 0.0) / dz[..., : k - 1]).max(axis=-1)
cn_col[: w.shape[0] // 2] = 0.0  # restrict to the strong (positive-wind) columns
i, j = np.unravel_index(np.argmax(cn_col), cn_col.shape)
lev = np.arange(k)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

ax = axes[0]
ax.plot(fx["ppm_adv"][i, j], lev, "s", color="k", ms=6, mfc="none", label="Isca")
ax.plot(adv[i, j], lev, "-o", color="tab:blue", ms=4, label="jsca")
ax.axvline(0, color="0.7", lw=0.8)
ax.invert_yaxis()
ax.set_xlabel("vertical advective tendency dr/dt (1/s)")
ax.set_ylabel("model level (0 = top)")
ax.set_title(f"PPM, strong-wind column (Courant>1) (i={i}, j={j})")
ax.legend(fontsize=8)

ax = axes[1]
half = w.shape[0] // 2
ax.scatter(fx["ppm_adv"][:half].ravel(), adv[:half].ravel(), s=12, alpha=0.4,
           color="tab:green", label="Courant<1 (both signs)")
ax.scatter(fx["ppm_adv"][half:].ravel(), adv[half:].ravel(), s=12, alpha=0.4,
           color="tab:red", label="Courant>1 (w>=0)")
lim = np.array([fx["ppm_adv"].min(), fx["ppm_adv"].max()]) * 1.05
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("Isca tendency (1/s)")
ax.set_ylabel("jsca tendency (1/s)")
ax.set_title("all grid points")
ax.legend(fontsize=8)

maxabs = np.abs(adv - fx["ppm_adv"]).max()
fig.suptitle(
    "PPM vertical advection (finite_volume_parabolic, incl. Courant>1): jsca vs Isca  "
    f"— max |Δ| = {maxabs:.0e} (machine precision)",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "docs" / "figures" / "ppm_advection.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
