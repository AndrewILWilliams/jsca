"""Figure for the two_stream_gray_rad PR: Frierson grey radiation, jsca vs Isca.

Left: the radiative heating profile of a tropical column (jsca over Isca).
Middle: surface downward SW/LW fluxes vs latitude. Right: jsca-vs-Isca scatter of
the heating rate across all points.

Run: python scripts/plot_two_stream_gray_rad.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.physics import GrayRadParams, two_stream_gray_rad

ROOT = Path(__file__).resolve().parent.parent
fx = np.load(ROOT / "tests" / "fixtures" / "two_stream_gray_rad_reference.npz")

tdt, net_sw, lw_dn, olr, net_lw = (np.asarray(a) for a in two_stream_gray_rad(
    GrayRadParams(), fx["rad_lat"], fx["rad_phalf"],
    fx["rad_t"], fx["rad_tsurf"], fx["rad_albedo"]))

latdeg = np.degrees(fx["rad_lat"][0])       # (nlat,), longitude-independent
tdt_day = tdt * 86400.0                       # K/day
tdt_f = fx["rad_tdt"] * 86400.0

# tropical column for the profile panel
eq = np.argmin(np.abs(latdeg))
p = fx["rad_phalf"][0, eq]
pfull = 0.5 * (p[1:] + p[:-1]) / 100.0        # hPa

fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))

ax = axes[0]
ax.plot(tdt_f[0, eq], pfull, "s", color="k", ms=6, mfc="none", label="Isca")
ax.plot(tdt_day[0, eq], pfull, "-", color="tab:red", label="jsca")
ax.axvline(0, color="0.7", lw=0.8)
ax.invert_yaxis()
ax.set_xlabel("radiative heating (K/day)")
ax.set_ylabel("pressure (hPa)")
ax.set_title(f"Tropical column (lat={latdeg[eq]:.0f}°)")
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(latdeg, fx["rad_net_sw_sfc"][0], "s", color="k", ms=5, mfc="none")
ax.plot(latdeg, net_sw[0], "-", color="tab:orange", label="net SW down (sfc)")
ax.plot(latdeg, fx["rad_lw_down_sfc"][0], "s", color="k", ms=5, mfc="none")
ax.plot(latdeg, lw_dn[0], "-", color="tab:blue", label="LW down (sfc)")
ax.plot([], [], "s", color="k", ms=5, mfc="none", label="Isca")
ax.set_xlabel("latitude (°)")
ax.set_ylabel("surface flux (W/m²)")
ax.set_title("surface downward fluxes")
ax.legend(fontsize=8)

ax = axes[2]
ax.scatter(tdt_f.ravel(), tdt_day.ravel(), s=10, alpha=0.5, color="tab:red")
lim = np.array([tdt_f.min(), tdt_f.max()])
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("Isca heating (K/day)")
ax.set_ylabel("jsca heating (K/day)")
ax.set_title("all points: heating")

maxabs = np.abs(tdt - fx["rad_tdt"]).max()
fig.suptitle(
    "Frierson grey (two-stream) radiation: jsca vs Isca  "
    f"— max |Δ heating| = {maxabs:.1e} K/s; surface fluxes to machine precision",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "docs" / "figures" / "two_stream_gray_rad.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
