"""Figure for the monin_obukhov PR: surface-layer similarity, jsca vs Isca.

Left: the momentum drag coefficient vs bulk Richardson number — enhanced in
unstable air, collapsing to drag_min as the flow approaches the critical
Richardson number in stable air. Right: jsca-vs-Isca scatter of the drag
coefficients and friction velocity across all surface-layer states.

Run: python scripts/plot_monin_obukhov.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.physics import MOParams, mo_drag

ROOT = Path(__file__).resolve().parent.parent
fx = np.load(ROOT / "tests" / "fixtures" / "monin_obukhov_reference.npz")

p = MOParams()
drag_m, drag_t, drag_q, u_star, b_star = (np.asarray(a) for a in mo_drag(
    p, fx["mo_pt"], fx["mo_pt0"], fx["mo_z"], fx["mo_z0"], fx["mo_zt"],
    fx["mo_zq"], fx["mo_speed"]))

# reconstruct the bulk Richardson number for the x-axis
delta_b = p.grav * (fx["mo_pt0"] - fx["mo_pt"]) / fx["mo_pt0"]
rich = -fx["mo_z"] * delta_b / (fx["mo_speed"] ** 2 + p.small)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

ax = axes[0]
order = np.argsort(rich)
ax.plot(rich[order], fx["mo_drag_m"][order], "s", color="k", ms=5, mfc="none",
        label="Isca")
ax.plot(rich[order], drag_m[order], ".", color="tab:blue", label="jsca")
ax.axhline(p.drag_min, color="tab:red", ls="--", lw=0.8, label="drag_min")
ax.axvline(0.95 * p.rich_crit, color="0.6", ls=":", lw=0.8, label="0.95·Ri$_c$")
ax.set_yscale("log")
ax.set_xlabel("bulk Richardson number")
ax.set_ylabel("momentum drag coefficient $c_{d,m}$")
ax.set_title("drag vs stability")
ax.legend(fontsize=8)

ax = axes[1]
for arr, ref, c, lab in [(drag_m, "mo_drag_m", "tab:blue", "$c_{d,m}$"),
                         (drag_t, "mo_drag_t", "tab:orange", "$c_{d,t}$"),
                         (drag_q, "mo_drag_q", "tab:green", "$c_{d,q}$")]:
    ax.scatter(fx[ref], arr, s=14, alpha=0.6, color=c, label=lab)
lim = np.array([0, max(fx["mo_drag_m"].max(), fx["mo_drag_q"].max()) * 1.05])
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("Isca drag coefficient")
ax.set_ylabel("jsca drag coefficient")
ax.set_title("all points: drag")
ax.legend(fontsize=8)

ax = axes[2]
ax.scatter(fx["mo_u_star"], u_star, s=16, alpha=0.6, color="tab:purple")
lim = np.array([fx["mo_u_star"].min(), fx["mo_u_star"].max()])
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("Isca $u_*$ (m/s)")
ax.set_ylabel("jsca $u_*$ (m/s)")
ax.set_title("all points: friction velocity")

maxrel = np.max(np.abs(drag_m - fx["mo_drag_m"]) / np.maximum(fx["mo_drag_m"], 1e-30))
fig.suptitle(
    "Monin-Obukhov surface-layer similarity: jsca vs Isca  "
    f"— max rel. drag diff = {maxrel:.1e} (machine precision)",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "docs" / "figures" / "monin_obukhov.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
