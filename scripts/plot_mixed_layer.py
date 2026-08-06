"""Figure for the mixed_layer PR: slab-ocean surface energy balance, jsca vs Isca.

Left: the sea-surface-temperature increment per step (jsca vs Isca) across the
4x6 test grid — the closed implicit balance of the surface fluxes against the
slab-ocean heat capacity. Right: jsca-vs-Isca scatter of the three step outputs
(SST update and the corrected lowest-level T/q increments the vert_diff up sweep
consumes).

Run: python scripts/plot_mixed_layer.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.physics import MixedLayerParams, mixed_layer_step
from jsca.physics.vert_diff import TriSurf

ROOT = Path(__file__).resolve().parent.parent
fx = np.load(ROOT / "tests" / "fixtures" / "mixed_layer_reference.npz")

DT = 720.0
z = np.zeros_like(fx["ml_t_surf_in"])
tri = TriSurf(
    e=z, f_t=z, f_q=z,
    delta_t=fx["ml_delta_t_in"], delta_q=fx["ml_delta_q_in"],
    mu_delt_n=fx["ml_dtmass"], nu_n=z, e_n1=z, dflux=fx["ml_dflux_t"],
)
params = MixedLayerParams(depth=2.5, albedo=0.31, evaporation=True)
t_surf_new, delta_t_surf, tri_out = mixed_layer_step(
    params, fx["ml_t_surf_in"], fx["ml_flux_t"], fx["ml_flux_q"], fx["ml_flux_r"],
    fx["ml_net_sw"], fx["ml_lw_down"], fx["ml_dhdt_surf"], fx["ml_dedt_surf"],
    fx["ml_drdt_surf"], fx["ml_dhdt_atm"], fx["ml_dedq_atm"], tri, DT)
t_surf_new = np.asarray(t_surf_new)
delta_t = np.asarray(tri_out.delta_t)
delta_q = np.asarray(tri_out.delta_q)

dts_jsca = (t_surf_new - fx["ml_t_surf_in"]).ravel()
dts_isca = (fx["ml_t_surf_out"] - fx["ml_t_surf_in"]).ravel()

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

ax = axes[0]
idx = np.arange(dts_jsca.size)
ax.plot(idx, dts_isca * 1e3, "s", color="k", ms=6, mfc="none", label="Isca")
ax.plot(idx, dts_jsca * 1e3, ".", color="tab:red", ms=8, label="jsca")
ax.axhline(0, color="0.7", lw=0.8)
ax.set_xlabel("grid point")
ax.set_ylabel("SST increment ΔTs (mK / step)")
ax.set_title("slab-ocean SST update per 720 s step")
ax.legend(fontsize=8)

ax = axes[1]
for arr, ref, c, lab, sc in [
        (t_surf_new - fx["ml_t_surf_in"], fx["ml_t_surf_out"] - fx["ml_t_surf_in"],
         "tab:red", "ΔTs (mK)", 1e3),
        (delta_t, fx["ml_delta_t_out"], "tab:orange", "δT low level (mK)", 1e3),
        (delta_q, fx["ml_delta_q_out"], "tab:green", "δq low level (mg/kg)", 1e6)]:
    ax.scatter(np.asarray(ref).ravel() * sc, np.asarray(arr).ravel() * sc,
               s=18, alpha=0.6, color=c, label=lab)
lo = min(dts_isca.min() * 1e3, (fx["ml_delta_t_out"]).min() * 1e3)
hi = max(dts_isca.max() * 1e3, (fx["ml_delta_t_out"]).max() * 1e3)
lim = np.array([lo, hi]) * 1.1
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("Isca output")
ax.set_ylabel("jsca output")
ax.set_title("all step outputs (scaled)")
ax.legend(fontsize=8)

maxabs = max(np.abs(t_surf_new - fx["ml_t_surf_out"]).max(),
             np.abs(delta_t - fx["ml_delta_t_out"]).max(),
             np.abs(delta_q - fx["ml_delta_q_out"]).max())
fig.suptitle(
    "Slab-ocean surface energy balance (mixed_layer, implicit SST update): jsca vs Isca  "
    f"— max |Δ| = {maxabs:.0e} (machine precision)",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "docs" / "figures" / "mixed_layer.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
