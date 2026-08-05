"""Build the pinned Isca Held-Suarez reference climatology from a benchmark run.

Reads the monthly ``atmos_monthly.nc`` files produced by
``bench/run_isca_held_suarez.py``, discards the first ``--spinup`` months as
spin-up, time- and zonal-averages the rest, and writes:

  - ``baseline/reference/hs_isca_reference.npz`` — zonal-mean u, T, v, ps and
    the vertical coordinate (the validation target for jsca's Held-Suarez).
  - ``docs/figures/hs_isca_reference.png`` — the canonical HS94 diagnostic
    (zonal-mean zonal wind + temperature).

Usage:
    python baseline/make_hs_reference.py \
        --data $GFDL_DATA/held_suarez_bench --spinup 4

Requires xarray + netCDF4 (analysis only — not a model dependency).
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with run00NN/atmos_monthly.nc")
    ap.add_argument("--spinup", type=int, default=4, help="months to discard")
    ap.add_argument("--npz", default="baseline/reference/hs_isca_reference.npz")
    ap.add_argument("--fig", default="docs/figures/hs_isca_reference.png")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.data}/run00*/atmos_monthly.nc"))[args.spinup:]
    print(f"averaging {len(files)} months: "
          f"{[os.path.basename(os.path.dirname(f)) for f in files]}")

    # load each month, zonal-mean, then average across months (no dask needed)
    us, ts, vs, pss = [], [], [], []
    for f in files:
        d = xr.open_dataset(f, decode_times=False)
        us.append(d["ucomp"].isel(time=0).mean("lon").values)  # (pfull, lat)
        ts.append(d["temp"].isel(time=0).mean("lon").values)
        vs.append(d["vcomp"].isel(time=0).mean("lon").values)
        pss.append(d["ps"].isel(time=0).mean("lon").values / 100.0)  # hPa
    ds = xr.open_dataset(files[-1], decode_times=False)
    lat = ds["lat"].values
    pfull = ds["pfull"].values  # hPa
    u = np.mean(us, axis=0)  # (pfull, lat)
    t = np.mean(ts, axis=0)
    v = np.mean(vs, axis=0)
    ps = np.mean(pss, axis=0)

    jmax = np.unravel_index(np.nanargmax(u), u.shape)
    print(f"jet: max u={u.max():.1f} m/s at lat={lat[jmax[1]]:.0f}, "
          f"p={pfull[jmax[0]]:.0f} hPa")
    print(f"max T={t.max():.1f} K, lowest-level mean T={t[-1].mean():.1f} K")

    np.savez_compressed(
        args.npz,
        lat=lat, pfull=pfull, u_zm=u, t_zm=t, v_zm=v, ps_zm=ps,
        bk=ds["bk"].values, pk=ds["pk"].values, nmonths=len(files),
    )
    print(f"saved {args.npz}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    lev = np.arange(-30, 36, 4)
    c = ax[0].contourf(lat, pfull, u, levels=lev, cmap="RdBu_r", extend="both")
    ax[0].contour(lat, pfull, u, levels=[0], colors="k", linewidths=0.6)
    plt.colorbar(c, ax=ax[0], label="m/s")
    ax[0].invert_yaxis()
    ax[0].set_ylabel("pressure (hPa)")
    ax[0].set_xlabel("latitude")
    ax[0].set_title("zonal-mean zonal wind")

    tl = np.arange(190, 316, 10)
    c2 = ax[1].contourf(lat, pfull, t, levels=tl, cmap="viridis", extend="both")
    ax[1].contour(lat, pfull, t, levels=tl, colors="k", linewidths=0.3)
    plt.colorbar(c2, ax=ax[1], label="K")
    ax[1].invert_yaxis()
    ax[1].set_ylabel("pressure (hPa)")
    ax[1].set_xlabel("latitude")
    ax[1].set_title("zonal-mean temperature")

    fig.suptitle(
        f"Isca Held-Suarez reference climatology (T42L25, {len(files) * 30}-day mean)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(args.fig, dpi=120)
    print(f"saved {args.fig}")


if __name__ == "__main__":
    main()
