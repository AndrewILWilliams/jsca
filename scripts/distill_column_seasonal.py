"""Distill the raw Isca do_seasonal single-column reference (NetCDF) into a small
numpy `.npz` for the CI regression test (`tests/test_column_seasonal_vs_isca.py`).

CI installs only `.[dev]` (no xarray/netcdf), so the committed reference the test
loads must be numpy-only. This reads the archived Isca seasonal run
(`baseline/reference/column_scm_isca_seasonal.nc`, produced by
`scripts/run_isca_column_seasonal.py` with `two_stream_gray_rad_nml:do_seasonal=.true.`)
and writes the daily-mean fields the test needs, including `swdn_toa` (the
seasonally varying TOA insolation this run exists to validate).

Usage: python scripts/distill_column_seasonal.py
"""
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
src = ROOT / "baseline" / "reference" / "column_scm_isca_seasonal.nc"
dst = ROOT / "baseline" / "reference" / "column_scm_isca_seasonal.npz"

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
    lon_deg=np.asarray(ds["lon"].values, dtype=float).ravel(),
    n_days=np.array(n),
    # the seasonal cycle these arrays exist to validate: daily-mean TOA insolation
    swdn_toa=col("swdn_toa"),              # W/m^2
    # daily-mean time series (single column)
    t_surf=col("t_surf"),
    precip=col("precipitation"),          # kg/m^2/s
    ps=col("ps"),
    # daily-mean profiles (n_days, K), level-last (k=0 top .. K-1 surface)
    temp=np.asarray(ds["temp"].values).reshape(n, K, -1)[:, :, 0],
    sphum=np.asarray(ds["sphum"].values).reshape(n, K, -1)[:, :, 0],
)
print(f"wrote {dst}  ({K} levels, {n} days, "
      f"lat={float(ds['lat'].values.ravel()[0]):.3f}, "
      f"swdn_toa {float(col('swdn_toa')[0]):.1f}->{float(col('swdn_toa')[-1]):.1f} W/m2)")
