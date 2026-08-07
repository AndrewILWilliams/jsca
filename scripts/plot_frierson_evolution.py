"""Temporal evolution of Frierson global means -- jsca vs Isca.

Overlays the spin-up trajectories of the area-weighted global means (mass-weighted
T, precip mm/day, t_surf) for jsca and, if available, the real Isca run. Both
runs start from the *same* initial condition (Isca's Frierson IC), so the
trajectories are directly comparable -- this is the diagnostic behind the
"is the cold bias just slow equilibration?" question in
``docs/frierson_climatology.md``.

References (in ``baseline/reference/``):
  * ``frierson_jsca_evolution_t42x64.npz`` -- jsca (``day``, ``gm_T``,
    ``gm_precip``, ``gm_tsurf``, ``spinup_end_day``);
  * ``frierson_isca_evolution_t42.npz``   -- Isca, same keys (optional).

Usage: ``python scripts/plot_frierson_evolution.py [jsca.npz] [out.png]``
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ref = ROOT / "baseline" / "reference"
jsrc = Path(sys.argv[1]) if len(sys.argv) > 1 else ref / "frierson_jsca_evolution_t42x64.npz"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    ROOT / "docs" / "figures" / "frierson_spinup_evolution.png"
isrc = ref / "frierson_isca_evolution_t42.npz"

J = np.load(jsrc)
I = np.load(isrc) if isrc.exists() else None
spinup_end = float(J["spinup_end_day"]) if "spinup_end_day" in J else None

fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
panels = [("gm_T", "global-mean temperature", "K (mass-weighted)"),
          ("gm_precip", "global-mean precipitation", "mm/day"),
          ("gm_tsurf", "global-mean surface temperature", "K")]
for ax, (key, title, ylab) in zip(axes, panels):
    ax.plot(J["day"], J[key], color="C3", lw=1.6, label="jsca")
    if I is not None:
        ax.plot(I["day"], I[key], color="k", lw=1.6, ls="--", label="Isca")
    if spinup_end is not None:
        ax.axvline(spinup_end, color="grey", ls=":", lw=1, alpha=0.6)
        ax.text(spinup_end, ax.get_ylim()[1], " jsca avg window", va="top", fontsize=7, alpha=0.7)
    ax.set_xlabel("simulated day")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

fig.suptitle("Frierson global-mean evolution from the same initial condition "
             "(T42, 64x128): jsca vs Isca", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out, dpi=120, bbox_inches="tight")
msg = f"wrote {out} (jsca day {J['day'][0]:.0f}-{J['day'][-1]:.0f}"
msg += f", Isca day {I['day'][0]:.0f}-{I['day'][-1]:.0f})" if I is not None else ", no Isca ref)"
print(msg)
