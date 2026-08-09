"""Distill the raw Isca single-column reference (NetCDF) into a small numpy `.npz`
for the CI regression test (`tests/test_column_vs_isca.py`).

CI installs only `.[dev]` (no xarray/netcdf), so the committed reference the test
loads must be numpy-only. This reads the archived Isca run
(`baseline/reference/column_scm_isca_daily_t264.nc`, produced by
`scripts/run_isca_column_reference.py`) and writes the daily-mean fields the test
needs. Re-run this whenever the Isca reference is regenerated.

Usage: python scripts/distill_column_reference.py
"""
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
src = ROOT / "baseline" / "reference" / "column_scm_isca_daily_t264.nc"
dst = ROOT / "baseline" / "reference" / "column_scm_isca_t264.npz"

ds = xr.open_dataset(src, decode_times=False)
n = ds.sizes["time"]
K = ds.sizes["pfull"]
col = lambda v: np.asarray(ds[v].values).reshape(n, -1)[:, 0]  # noqa: E731

np.savez_compressed(
    dst,
    pk=np.asarray(ds["pk"].values, dtype=float),
    bk=np.asarray(ds["bk"].values, dtype=float),
    pfull_hpa=np.asarray(ds["pfull"].values, dtype=float),
    lat_deg=np.asarray(ds["lat"].values, dtype=float).ravel(),
    n_days=np.array(n),
    # daily-mean time series (single column)
    t_surf=col("t_surf"),
    precip=col("precipitation"),          # kg/m^2/s
    ps=col("ps"),
    # daily-mean profiles (n_days, K), level-last (k=0 top .. K-1 surface)
    temp=np.asarray(ds["temp"].values).reshape(n, K, -1)[:, :, 0],
    sphum=np.asarray(ds["sphum"].values).reshape(n, K, -1)[:, :, 0],
)
print(f"wrote {dst}  ({K} levels, {n} days, lat={float(ds['lat'].values.ravel()[0]):.3f})")
