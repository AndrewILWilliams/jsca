"""Extract the Isca global-mean spin-up time series from an Isca run's daily output.

Reads a pinned-Isca ``frierson_test_case`` ``atmos_daily.nc`` and writes
``baseline/reference/frierson_isca_evolution_t42.npz`` with per-day area-weighted
global means (``day``, ``gm_T``, ``gm_precip``, ``gm_tsurf``) computed with the
*same* weighting as the jsca driver: mass-weighted vertical mean of ``temp`` over
the Frierson sigma layers (``sum_k T_k * dbk``), then cos-lat area mean; precip in
mm/day. This is the Isca curve overlaid by ``plot_frierson_evolution.py``.

Usage: ``python scripts/extract_isca_evolution.py /path/to/atmos_daily.nc [out.npz]``
The netCDF is produced by the run recipe in
``fortran_instrumentation/frierson_step_recipe.md``.
"""
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

from jsca.model.frierson import FRIERSON_BK

src = sys.argv[1]
_ref = Path(__file__).resolve().parent.parent / "baseline" / "reference"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else _ref / "frierson_isca_evolution_t42.npz"

d = nc.Dataset(src)
lat = d.variables["lat"][:]
day = d.variables["time"][:]
temp = d.variables["temp"][:]            # (time, pfull, lat, lon)
precip = d.variables["precipitation"][:]  # (time, lat, lon), kg/m^2/s
tsurf = d.variables["t_surf"][:]          # (time, lat, lon)
d.close()

w = np.cos(np.deg2rad(lat))
Wsum = w.sum()
dbk = FRIERSON_BK[1:] - FRIERSON_BK[:-1]   # sigma-layer thickness, sums to 1


def gm2d(f):                               # (time, lat, lon) -> (time,)
    return (w[None, :, None] * f).sum(axis=(1, 2)) / (Wsum * f.shape[2])


gm_T = gm2d((temp * dbk[None, :, None, None]).sum(axis=1))
gm_precip = gm2d(precip) * 86400.0
gm_tsurf = gm2d(tsurf)
np.savez(out, day=np.asarray(day), gm_T=gm_T, gm_precip=gm_precip, gm_tsurf=gm_tsurf)
print(f"wrote {out} (day {day[0]:.1f}-{day[-1]:.1f}; "
      f"gm_T {gm_T[0]:.1f}->{gm_T[-1]:.1f}, gm_tsurf {gm_tsurf[0]:.1f}->{gm_tsurf[-1]:.1f})")
