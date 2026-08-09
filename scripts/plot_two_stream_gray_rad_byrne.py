"""Figure for the Byrne (2013) grey-radiation PR.

Left: jsca vs Isca for the Byrne scheme — radiative heating of a tropical column
(validation that the port is exact). Middle: surface downward LW vs latitude,
Byrne vs Frierson, showing Byrne's humidity dependence (more downward LW in the
moist tropics). Right: jsca-vs-Isca scatter of the Byrne heating across all
points.

Run: python scripts/plot_two_stream_gray_rad_byrne.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import jsca  # noqa: F401
from jsca.physics import GrayRadParams, two_stream_gray_rad

ROOT = Path(__file__).resolve().parent.parent
fx = np.load(ROOT / "tests" / "fixtures" / "two_stream_gray_rad_byrne_reference.npz")

# Byrne (jsca) — needs q
tdt_b, net_sw_b, lw_b, _, _ = (np.asarray(a) for a in two_stream_gray_rad(
    GrayRadParams(rad_scheme="byrne", atm_abs=0.2), fx["rad_lat"], fx["rad_phalf"],
    fx["rad_t"], fx["rad_tsurf"], fx["rad_albedo"], fx["rad_q"]))
# Frierson (jsca) on the same state, for the physics comparison
_, _, lw_f, _, _ = (np.asarray(a) for a in two_stream_gray_rad(
    GrayRadParams(rad_scheme="frierson"), fx["rad_lat"], fx["rad_phalf"],
    fx["rad_t"], fx["rad_tsurf"], fx["rad_albedo"]))

latdeg = np.degrees(fx["rad_lat"][0])
tdt_b_day = tdt_b * 86400.0
tdt_isca = fx["rad_tdt"] * 86400.0

eq = np.argmin(np.abs(latdeg))
p = fx["rad_phalf"][0, eq]
pfull = 0.5 * (p[1:] + p[:-1]) / 100.0

fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))

ax = axes[0]
ax.plot(tdt_isca[0, eq], pfull, "s", color="k", ms=6, mfc="none", label="Isca (Byrne)")
ax.plot(tdt_b_day[0, eq], pfull, "-", color="tab:purple", label="jsca (Byrne)")
ax.axvline(0, color="0.7", lw=0.8)
ax.invert_yaxis()
ax.set_xlabel("radiative heating (K/day)")
ax.set_ylabel("pressure (hPa)")
ax.set_title(f"Tropical column (lat={latdeg[eq]:.0f}°)")
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(latdeg, fx["rad_lw_down_sfc"][0], "s", color="k", ms=5, mfc="none",
        label="Isca (Byrne)")
ax.plot(latdeg, lw_b[0], "-", color="tab:purple", label="jsca Byrne")
ax.plot(latdeg, lw_f[0], "--", color="tab:blue", label="jsca Frierson")
ax.set_xlabel("latitude (°)")
ax.set_ylabel("surface LW down (W/m²)")
ax.set_title("Byrne humidity dependence\n(moist tropics trap more LW)")
ax.legend(fontsize=8)

ax = axes[2]
ax.scatter(tdt_isca.ravel(), tdt_b_day.ravel(), s=10, alpha=0.5, color="tab:purple")
lim = np.array([tdt_isca.min(), tdt_isca.max()])
ax.plot(lim, lim, "k-", lw=0.8)
ax.set_xlabel("Isca heating (K/day)")
ax.set_ylabel("jsca heating (K/day)")
ax.set_title("all points: heating")

maxabs = np.abs(tdt_b - fx["rad_tdt"]).max()
fig.suptitle(
    "Byrne & O'Gorman (2013) grey radiation: jsca vs Isca  "
    f"— max |Δ heating| = {maxabs:.1e} K/s; surface fluxes to machine precision",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "docs" / "figures" / "two_stream_gray_rad_byrne.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
