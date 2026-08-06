"""Figure for the damping_driver PR: top-of-model Rayleigh sponge, jsca vs Isca.

Left: the sponge's wind-tendency profile for one column (jsca over Isca) — zero
through the troposphere, ramping up sharply toward the model lid as the pressure
drops below sponge_pbottom. Right: jsca-vs-Isca scatter of all three outputs
(u/v drag and frictional heating).

Run: python scripts/plot_damping.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.physics import damping_driver_init, rayleigh_sponge

ROOT = Path(__file__).resolve().parent.parent
fx = np.load(ROOT / "tests" / "fixtures" / "damping_reference.npz")

DT = 720.0
params = damping_driver_init(fx["dd_pref"], trayfric=-0.25,
                             sponge_pbottom=5000.0, do_conserve_energy=True)
udt, vdt, tdt = rayleigh_sponge(params, DT, fx["dd_pfull"], fx["dd_u"], fx["dd_v"])
udt, vdt, tdt = (np.asarray(a) for a in (udt, vdt, tdt))

# column with the strongest drag
i, j = np.unravel_index(np.argmin(udt.min(axis=-1)), udt.shape[:-1])
p_col = fx["dd_pfull"][i, j] / 100.0    # hPa

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

ax = axes[0]
ax.plot(fx["dd_udt"][i, j] * 86400, p_col, "s", color="k", ms=6, mfc="none", label="Isca")
ax.plot(udt[i, j] * 86400, p_col, "-o", color="tab:blue", ms=4, label="jsca")
ax.axhline(50.0, color="tab:red", lw=0.8, ls="--", label="sponge_pbottom (50 hPa)")
ax.axvline(0, color="0.7", lw=0.8)
ax.set_ylim(p_col.max(), 0)   # pressure decreasing upward
ax.set_xlabel("zonal-wind tendency du/dt (m/s/day)")
ax.set_ylabel("pressure (hPa)")
ax.set_title(f"Rayleigh sponge profile (i={i}, j={j})")
ax.legend(fontsize=8)

ax = axes[1]
for arr, ref, c, lab, sc in [
        (udt, "dd_udt", "tab:blue", "du/dt (m/s/day)", 86400),
        (vdt, "dd_vdt", "tab:cyan", "dv/dt (m/s/day)", 86400),
        (tdt, "dd_tdt", "tab:orange", "dT/dt ×10 (K/day)", 86400 * 10)]:
    ax.scatter(fx[ref].ravel() * sc, arr.ravel() * sc, s=14, alpha=0.5, color=c, label=lab)
lim = np.array([-1, 1]) * abs(fx["dd_udt"].min()) * 86400 * 1.1
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("Isca tendency")
ax.set_ylabel("jsca tendency")
ax.set_title("all points")
ax.legend(fontsize=8)

maxabs = max(np.abs(udt - fx["dd_udt"]).max(), np.abs(tdt - fx["dd_tdt"]).max())
fig.suptitle(
    "Top-of-model Rayleigh sponge (damping_driver): jsca vs Isca  "
    f"— max |Δ| = {maxabs:.0e} (machine precision)",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "docs" / "figures" / "damping.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
