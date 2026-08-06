"""Figure for the vert_diff PR: implicit vertical diffusion, jsca vs Isca.

Left: the momentum and temperature diffusion tendencies for one column (jsca over
Isca) — concentrated in the boundary layer where the surface stress and
diffusivity act. Right: jsca-vs-Isca scatter of all four tendencies.

Run: python scripts/plot_vert_diff.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.physics import vert_diff_down, vert_diff_up

ROOT = Path(__file__).resolve().parent.parent
fx = np.load(ROOT / "tests" / "fixtures" / "vert_diff_reference.npz")

DELT = 720.0
dt_u, dt_v, diss, tri = vert_diff_down(
    DELT, fx["vd_u"], fx["vd_v"], fx["vd_t"], fx["vd_q"],
    fx["vd_diff_m"], fx["vd_diff_t"], fx["vd_p_half"], fx["vd_p_full"],
    fx["vd_z_full"], fx["vd_tau_u"], fx["vd_tau_v"], fx["vd_dtau_du"], fx["vd_dtau_dv"])
dt_t, dt_q = vert_diff_up(DELT, tri)
dt_u, dt_v, dt_t, dt_q = (np.asarray(a) for a in (dt_u, dt_v, dt_t, dt_q))

z = fx["vd_z_full"]
i, j = np.unravel_index(np.argmax(np.abs(dt_u).max(axis=-1)), dt_u.shape[:-1])

fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))

ax = axes[0]
ax.plot(fx["vd_dt_u"][i, j] * 86400, z[i, j], "s", color="k", ms=5, mfc="none",
        label="Isca")
ax.plot(dt_u[i, j] * 86400, z[i, j], "-o", color="tab:blue", ms=3, label="jsca")
ax.axvline(0, color="0.7", lw=0.8)
ax.set_ylim(0, 2500)
ax.set_xlabel("momentum tendency du/dt (m/s/day)")
ax.set_ylabel("height (m)")
ax.set_title(f"momentum diffusion (i={i}, j={j})")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(fx["vd_dt_t"][i, j] * 86400, z[i, j], "s", color="k", ms=5, mfc="none",
        label="Isca")
ax.plot(dt_t[i, j] * 86400, z[i, j], "-o", color="tab:orange", ms=3, label="jsca")
ax.axvline(0, color="0.7", lw=0.8)
ax.set_ylim(0, 2500)
ax.set_xlabel("temperature tendency dT/dt (K/day)")
ax.set_ylabel("height (m)")
ax.set_title("heat diffusion + friction")
ax.legend(fontsize=8)

ax = axes[2]
for arr, ref, c, lab, sc in [
        (dt_u, "vd_dt_u", "tab:blue", "du/dt", 86400),
        (dt_v, "vd_dt_v", "tab:cyan", "dv/dt", 86400),
        (dt_t, "vd_dt_t", "tab:orange", "dT/dt", 86400),
        (dt_q, "vd_dt_q", "tab:green", "dq/dt ×1e3", 86400 * 1e3)]:
    ax.scatter(fx[ref].ravel() * sc, arr.ravel() * sc, s=10, alpha=0.5, color=c, label=lab)
lim = np.array([-1, 1]) * max(abs(fx["vd_dt_u"].min()), fx["vd_dt_u"].max()) * 86400 * 1.1
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("Isca tendency (/day)")
ax.set_ylabel("jsca tendency (/day)")
ax.set_title("all points")
ax.legend(fontsize=8)

maxabs = max(np.abs(dt_u - fx["vd_dt_u"]).max(), np.abs(dt_t - fx["vd_dt_t"]).max())
fig.suptitle(
    "Implicit vertical diffusion (vert_diff, down/up tridiagonal solve): jsca vs Isca  "
    f"— max |Δ| = {maxabs:.0e} (machine precision)",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "docs" / "figures" / "vert_diff.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
