"""Figure for the surface_flux PR: bulk ocean surface fluxes, jsca vs Isca.

Left: sensible-heat and evaporative fluxes vs the air-sea temperature contrast
(jsca over Isca). Right: jsca-vs-Isca scatter of the sensible, latent and
momentum fluxes across all surface-layer states.

Run: python scripts/plot_surface_flux.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.physics import surface_flux

ROOT = Path(__file__).resolve().parent.parent
fx = np.load(ROOT / "tests" / "fixtures" / "surface_flux_reference.npz")

n = fx["sf_t_atm"].shape[0]
r = surface_flux(
    fx["sf_t_atm"], fx["sf_q_atm"], fx["sf_u_atm"], fx["sf_v_atm"],
    fx["sf_p_atm"], fx["sf_z_atm"], fx["sf_p_surf"], fx["sf_t_surf"],
    np.zeros(n), np.zeros(n),
    fx["sf_rough_mom"], fx["sf_rough_heat"], fx["sf_rough_moist"],
    fx["sf_gust"], fx["sf_q_surf_in"])

dT = fx["sf_t_surf"] - fx["sf_t_atm"]           # air-sea temperature contrast
HLV = jsca.constants.HLV
lh_j = np.asarray(r.flux_q) * HLV                # latent heat flux (W/m^2)
lh_f = fx["sf_flux_q"] * HLV

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

ax = axes[0]
ax.scatter(dT, fx["sf_flux_t"], s=30, facecolor="none", edgecolor="k", label="Isca")
ax.scatter(dT, np.asarray(r.flux_t), s=12, color="tab:orange", label="jsca: sensible")
ax.axhline(0, color="0.7", lw=0.8)
ax.set_xlabel("air-sea contrast  $T_{surf}-T_{atm}$ (K)")
ax.set_ylabel("sensible heat flux (W/m²)")
ax.set_title("sensible heat vs air-sea contrast")
ax.legend(fontsize=8)

ax = axes[1]
ax.scatter(fx["sf_u_atm"], lh_f, s=30, facecolor="none", edgecolor="k", label="Isca")
ax.scatter(fx["sf_u_atm"], lh_j, s=12, color="tab:blue", label="jsca: latent")
ax.set_xlabel("wind speed (m/s)")
ax.set_ylabel("latent heat flux (W/m²)")
ax.set_title("evaporation vs wind")
ax.legend(fontsize=8)

ax = axes[2]
for arr, ref, c, lab in [(np.asarray(r.flux_t), "sf_flux_t", "tab:orange", "sensible"),
                         (lh_j, None, "tab:blue", "latent"),
                         (np.asarray(r.flux_u), "sf_flux_u", "tab:green", "stress ×1e3")]:
    if ref is None:
        ax.scatter(lh_f, arr, s=14, alpha=0.6, color=c, label=lab)
    elif lab.startswith("stress"):
        ax.scatter(fx[ref] * 1e3, arr * 1e3, s=14, alpha=0.6, color=c, label=lab)
    else:
        ax.scatter(fx[ref], arr, s=14, alpha=0.6, color=c, label=lab)
lim = np.array([-110, 110])
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("Isca flux (W/m² or Pa×1e3)")
ax.set_ylabel("jsca flux")
ax.set_title("all points")
ax.legend(fontsize=8)

maxrel = np.max(np.abs(np.asarray(r.flux_q) - fx["sf_flux_q"])
                / np.maximum(np.abs(fx["sf_flux_q"]), 1e-30))
fig.suptitle(
    "Bulk ocean surface fluxes (surface_flux, do_simple): jsca vs Isca  "
    f"— sensible/momentum machine-exact; latent max rel = {maxrel:.0e} (es table)",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "docs" / "figures" / "surface_flux.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
