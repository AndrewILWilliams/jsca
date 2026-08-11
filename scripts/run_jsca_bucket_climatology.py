"""Run a jsca bucket-model climatology at T21 and write the time-mean fields.

The jsca half of the Stage-4 bucket validation: integrate
:mod:`jsca.model.bucket_model` at T21 (32x64) over the **same** realistic-continents
``land.nc`` the Isca run uses, then write the time-mean 2-D fields for comparison.

Because the land breaks zonal symmetry, the comparison is on full ``(nlat, nlon)``
maps (not zonal means): ``bucket_depth``, ``precip`` (mm/day), ``t_surf``, plus
``temp``/``sphum`` and ``ps``.

Usage::

    python scripts/run_jsca_bucket_climatology.py --land /path/land.nc \\
        --spinup-days 300 --avg-days 360 --out baseline/reference/bucket_jsca_t21.npz

A T21 model-year is ~35 min on one CPU core; the reservoir needs a long spin-up to
approach equilibrium, so treat short runs as smoke checks.
"""
import argparse
import time
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

import jsca  # noqa: F401  (enables float64)
from jsca.model.bucket_model import (
    build_bucket_model,
    initial_state,
    integrate_climatology,
)

DT = 720.0
SECONDS_PER_DAY = 86400.0


def load_land(path, nlat, nlon):
    d = Dataset(path)
    land = np.asarray(d.variables["land_mask"][:], dtype=np.float64)
    if land.shape != (nlat, nlon):
        raise ValueError(f"land.nc is {land.shape}, expected ({nlat}, {nlon})")
    return land


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--land", required=True)
    ap.add_argument("--spinup-days", type=float, default=300.0)
    ap.add_argument("--avg-days", type=float, default=360.0)
    ap.add_argument("--nlat", type=int, default=32)
    ap.add_argument("--nlon", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    land = load_land(args.land, args.nlat, args.nlon)
    m = build_bucket_model(land, num_fourier=21, nlat=args.nlat, nlon=args.nlon)
    state = initial_state(m)

    steps_per_day = SECONDS_PER_DAY / DT
    spinup = int(round(args.spinup_days * steps_per_day))
    avg = int(round(args.avg_days * steps_per_day))
    print(f"T21 bucket run: spinup {spinup} steps, avg {avg} steps "
          f"(land frac {land.mean():.4f})")

    t0 = time.time()
    _state, clim = integrate_climatology(m, state, spinup, avg, cold_start=True)
    print(f"integration done in {(time.time() - t0) / 60:.1f} min")

    lat = np.degrees(m.base.dyn.transforms.lat) if hasattr(
        m.base.dyn.transforms, "lat") else np.arcsin(
        np.asarray(m.base.dyn.transforms.sin_lat)) * 180.0 / np.pi
    lon = np.arange(args.nlon) * 360.0 / args.nlon

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        lat=lat, lon=lon, land=land,
        bucket_depth=clim["bucket_depth"],
        precip=clim["precip"] * SECONDS_PER_DAY,   # kg/m^2/s -> mm/day
        t_surf=clim["t_surf"],
        temp=clim["temp"], sphum=clim["sphum"], ps=clim["ps"],
        spinup_days=args.spinup_days, avg_days=args.avg_days,
    )
    landm = land > 0.5
    print(f"wrote {args.out}")
    print(f"  land bucket_depth mean {clim['bucket_depth'][landm].mean():.3f} m, "
          f"precip global {(clim['precip'].mean() * SECONDS_PER_DAY):.2f} mm/day, "
          f"t_surf land {clim['t_surf'][landm].mean():.1f} K "
          f"ocean {clim['t_surf'][~landm].mean():.1f} K")


if __name__ == "__main__":
    main()
