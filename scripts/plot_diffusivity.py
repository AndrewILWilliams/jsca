"""Figure for the diffusivity PR: boundary-layer K-profile, jsca vs Isca.

Left/middle: the momentum and heat eddy-diffusivity profiles for a couple of
columns (jsca over Isca), with the diagnosed PBL top marked. Right:
jsca-vs-Isca scatter of all diffusivity values.

Run: python scripts/plot_diffusivity.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.physics import DiffusivityParams, diffusivity

ROOT = Path(__file__).resolve().parent.parent
fx = np.load(ROOT / "tests" / "fixtures" / "diffusivity_reference.npz")

k_m, k_t, h = (np.asarray(a) for a in diffusivity(
    DiffusivityParams(), fx["df_t"], np.zeros_like(fx["df_t"]),
    fx["df_u"], fx["df_v"], fx["df_z_full"], fx["df_z_half"],
    fx["df_u_star"], fx["df_b_star"]))

# half-level heights above surface (first K), matching k_m/k_t
z_surf = fx["df_z_half"][..., -1:]
zm = (fx["df_z_half"] - z_surf)[..., :-1]

# pick the two most-mixed columns (largest peak K_m), which show PBL structure
us = fx["df_u_star"]
peak = k_m.max(axis=-1)
flat = np.argsort(peak.ravel())[::-1]
cols = [np.unravel_index(flat[0], peak.shape),
        np.unravel_index(flat[len(flat) // 2], peak.shape)]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))

for ax, (kk, kk_f, title) in zip(axes[:2], [(k_m, "df_k_m", "momentum $K_m$"),
                                            (k_t, "df_k_t", "heat $K_t$")]):
    for (i, j), c in zip(cols, ["tab:blue", "tab:red"]):
        lab = f"u*={us[i, j]:.2f}, h={h[i, j]:.0f}m"
        ax.plot(fx[kk_f][i, j], zm[i, j], "s", color="k", ms=5, mfc="none")
        ax.plot(kk[i, j], zm[i, j], "-o", color=c, ms=3, label=lab)
        ax.axhline(h[i, j], color=c, ls=":", lw=0.8)
    ax.set_ylim(0, 800)
    ax.set_xlabel(f"{title} (m²/s)")
    ax.set_ylabel("height above surface (m)")
    ax.set_title(f"{title} profile\n(dotted = PBL top)")
    ax.plot([], [], "s", color="k", ms=5, mfc="none", label="Isca")
    ax.legend(fontsize=8)

ax = axes[2]
ax.scatter(fx["df_k_m"].ravel(), k_m.ravel(), s=12, alpha=0.5, color="tab:blue",
           label="$K_m$")
ax.scatter(fx["df_k_t"].ravel(), k_t.ravel(), s=12, alpha=0.5, color="tab:orange",
           label="$K_t$")
lim = np.array([0, max(fx["df_k_m"].max(), fx["df_k_t"].max()) * 1.05])
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("Isca diffusivity (m²/s)")
ax.set_ylabel("jsca diffusivity (m²/s)")
ax.set_title("all points")
ax.legend(fontsize=8)

maxrel = np.max(np.abs(k_m - fx["df_k_m"]) / np.maximum(fx["df_k_m"], 1e-30))
fig.suptitle(
    "Boundary-layer eddy diffusivity (diffusivity, do_simple): jsca vs Isca  "
    f"— max rel diff = {maxrel:.0e}; PBL depth exact",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "docs" / "figures" / "diffusivity.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
