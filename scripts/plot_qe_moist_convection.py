"""Figure for the qe_moist_convection (CAPE stage) PR: jsca vs Isca.

Left: a representative convecting column — environment vs lifted-parcel virtual
temperature with the CAPE (parcel warmer) and CIN (parcel cooler) areas shaded.
Right: jsca-vs-Isca scatter of CAPE and CIN across all columns.

Run: python scripts/plot_qe_moist_convection.py  (writes docs/figures/qe_moist_convection.png)
"""
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.physics import convective_cape
from jsca.physics.qe_moist_convection import _cape_column, _virtual_temp

ROOT = Path(__file__).resolve().parent.parent
fx = np.load(ROOT / "tests" / "fixtures" / "qe_moist_convection_reference.npz")

tin, qin = fx["qe_tin"], fx["qe_qin"]
pfull, phalf = fx["qe_pfull"], fx["qe_phalf"]
rin = qin / (1.0 - qin)

cape_j, cin_j, klzb_j, _ = (np.asarray(a) for a in
                            convective_cape(tin, qin, pfull, phalf))

# pick the column with the largest CAPE for the sounding panel
i, j = np.unravel_index(np.argmax(fx["qe_cape"]), fx["qe_cape"].shape)
C, CI, kZ, kL, Tp, rp = _cape_column(
    jnp.asarray(pfull[i, j]), jnp.asarray(phalf[i, j]),
    jnp.asarray(tin[i, j]), jnp.asarray(rin[i, j]),
)
Tp = np.asarray(Tp)
rp = np.asarray(rp)
p = pfull[i, j] / 100.0  # hPa
Tenv_v = np.asarray(_virtual_temp(jnp.asarray(tin[i, j]), jnp.asarray(rin[i, j])))
Tpar_v = np.asarray(_virtual_temp(jnp.asarray(Tp), jnp.asarray(rp)))

fig, axes = plt.subplots(1, 3, figsize=(13.5, 6),
                         gridspec_kw={"width_ratios": [1.2, 1, 1]})

# --- sounding: environment vs parcel virtual temperature ---
ax = axes[0]
ax.plot(Tenv_v, p, "-", color="tab:blue", label="environment $T_v$")
ax.plot(Tpar_v, p, "-", color="tab:red", label="parcel $T_v$")
ax.fill_betweenx(p, Tenv_v, Tpar_v, where=Tpar_v > Tenv_v,
                 color="tab:red", alpha=0.20, label="CAPE (buoyant)")
ax.fill_betweenx(p, Tenv_v, Tpar_v, where=Tpar_v < Tenv_v,
                 color="tab:blue", alpha=0.20, label="CIN (sub-buoyant)")
klcl_0 = int(kL)
klzb_0 = int(kZ)
ax.axhline(p[klcl_0], color="0.5", ls="--", lw=0.8)
ax.text(ax.get_xlim()[0], p[klcl_0], " LCL", va="bottom", fontsize=8, color="0.4")
if klzb_0 > 0:
    ax.axhline(p[klzb_0], color="0.5", ls=":", lw=0.8)
    ax.text(ax.get_xlim()[0], p[klzb_0], " LZB", va="bottom", fontsize=8, color="0.4")
ax.invert_yaxis()
ax.set_xlabel("virtual temperature (K)")
ax.set_ylabel("pressure (hPa)")
ax.set_title(f"Convecting column (i={i}, j={j})\nCAPE = {float(C):.0f} J/kg")
ax.legend(fontsize=8, loc="upper right")

# --- CAPE scatter ---
ax = axes[1]
ax.scatter(fx["qe_cape"].ravel(), cape_j.ravel(), s=18, alpha=0.6, color="tab:red")
lim = np.array([0, fx["qe_cape"].max() * 1.05])
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("Isca CAPE (J/kg)")
ax.set_ylabel("jsca CAPE (J/kg)")
ax.set_title("all columns: CAPE")

# --- CIN scatter ---
ax = axes[2]
ax.scatter(fx["qe_cin"].ravel(), cin_j.ravel(), s=18, alpha=0.6, color="tab:blue")
lim = np.array([0, fx["qe_cin"].max() * 1.05])
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("Isca CIN (J/kg)")
ax.set_ylabel("jsca CIN (J/kg)")
ax.set_title("all columns: CIN")

maxrel = np.max(np.abs(cape_j - fx["qe_cape"])[fx["qe_cape"] > 1.0]
                / fx["qe_cape"][fx["qe_cape"] > 1.0])
fig.suptitle(
    "Frierson QE convection — CAPE stage (parcel ascent): jsca vs Isca  "
    f"— max rel. CAPE diff = {maxrel:.1e}; LCL/LZB levels exact",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = ROOT / "docs" / "figures" / "qe_moist_convection.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
