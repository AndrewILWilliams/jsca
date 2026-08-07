"""Temporal evolution of jsca Frierson global means (spin-up + averaging window).

Reads a time-series ``.npz`` written by the climatology driver — chunk-mean,
area-weighted global means recorded every ~1.7 simulated days:

  ``day``, ``gm_T`` (mass-weighted global-mean temperature, K),
  ``gm_precip`` (global-mean precipitation, mm/day), ``gm_tsurf`` (K).

Plots the three series so the approach to equilibrium (and any residual drift)
is visible at a glance — the diagnostic behind the "is the cold bias just
incomplete equilibration?" question in ``docs/frierson_climatology.md``.

Usage: ``python scripts/plot_frierson_evolution.py [timeseries.npz] [out.png]``
Defaults read the committed ``baseline/reference/frierson_jsca_evolution_t42x64.npz``.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / "baseline" / "reference" / "frierson_jsca_evolution_t42x64.npz"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    ROOT / "docs" / "figures" / "frierson_spinup_evolution.png"

d = np.load(src)
day, gm_T, gm_P, gm_Ts = d["day"], d["gm_T"], d["gm_precip"], d["gm_tsurf"]
spinup_end = float(d["spinup_end_day"]) if "spinup_end_day" in d else None

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
panels = [
    (gm_T, "global-mean temperature", "K (mass-weighted)", "C3"),
    (gm_P, "global-mean precipitation", "mm/day", "C0"),
    (gm_Ts, "global-mean surface temperature", "K", "C2"),
]
for ax, (y, title, ylab, c) in zip(axes, panels):
    ax.plot(day, y, color=c, lw=1.5)
    if spinup_end is not None:
        ax.axvline(spinup_end, color="k", ls=":", lw=1, alpha=0.6)
        ax.text(spinup_end, ax.get_ylim()[1], " avg window", va="top", fontsize=8, alpha=0.7)
    ax.set_xlabel("simulated day")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    # annotate the last value + drift over the final 50 days
    late = day >= day[-1] - 50
    if late.sum() > 1:
        slope = np.polyfit(day[late], y[late], 1)[0]
        ax.text(0.98, 0.04, f"end {y[-1]:.2f}\nlast-50d drift {slope * 50:+.2f}/50d",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", fc="w", alpha=0.7))

fig.suptitle("jsca Frierson spin-up evolution (T42, 64x128, Isca-matched IC)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out, dpi=120, bbox_inches="tight")
print("wrote", out, f"({len(day)} points, day {day[0]:.1f}-{day[-1]:.1f})")
