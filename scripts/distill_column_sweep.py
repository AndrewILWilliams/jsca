"""Distill the Isca latitude-sweep NetCDF runs into one numpy `.npz` for the CI
regression test (`tests/test_column_vs_isca.py`). Numpy-only reference, same as
`distill_column_reference.py` but stacked over latitude.

Reads baseline/reference/column_scm_isca_lat{L}.nc (from run_isca_column_sweep.py)
and writes baseline/reference/column_scm_isca_sweep.npz.

Usage: python scripts/distill_column_sweep.py
"""
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
REFDIR = ROOT / "baseline" / "reference"
LATS = [0, 15, 30, 45, 60]

t_surf, precip, temp, sphum, lat_deg = [], [], [], [], []
pk = bk = pfull = None
for L in LATS:
    ds = xr.open_dataset(REFDIR / f"column_scm_isca_lat{L}.nc", decode_times=False)
    n, K = ds.sizes["time"], ds.sizes["pfull"]
    col = lambda v: np.asarray(ds[v].values).reshape(n, -1)[:, 0]  # noqa: E731
    t_surf.append(col("t_surf"))
    precip.append(col("precipitation"))
    temp.append(np.asarray(ds["temp"].values).reshape(n, K, -1)[:, :, 0])
    sphum.append(np.asarray(ds["sphum"].values).reshape(n, K, -1)[:, :, 0])
    lat_deg.append(float(ds["lat"].values.ravel()[0]))
    pk = np.asarray(ds["pk"].values, dtype=float)
    bk = np.asarray(ds["bk"].values, dtype=float)
    pfull = np.asarray(ds["pfull"].values, dtype=float)

dst = REFDIR / "column_scm_isca_sweep.npz"
np.savez_compressed(
    dst, pk=pk, bk=bk, pfull_hpa=pfull, lat_deg=np.asarray(lat_deg),
    n_days=np.array(temp[0].shape[0]),
    t_surf=np.stack(t_surf), precip=np.stack(precip),      # (nlat, n_days)
    temp=np.stack(temp), sphum=np.stack(sphum),            # (nlat, n_days, K)
)
print(f"wrote {dst}  lats={lat_deg}")
for i, L in enumerate(lat_deg):
    print(f"  lat {L:5.1f}: final SST {t_surf[i][-1]:.2f} K, "
          f"precip {precip[i][-1]*86400:.2f} mm/day")
