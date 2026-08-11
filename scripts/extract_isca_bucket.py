"""Extract the Isca bucket run's time-mean 2-D fields into a comparison ``.npz``.

Reads the monthly output of ``scripts/run_isca_bucket.py`` (``atmos_monthly.nc``
under each ``runNNNN/`` directory in ``$GFDL_DATA/<name>/``) and writes the
time-mean ``bucket_depth``, ``precip`` (mm/day) and ``t_surf`` maps over a
trailing averaging window, on the model's lat/lon grid — the Isca side of the
jsca-vs-Isca bucket comparison (mirrors ``extract_isca_evolution.py``).

Usage::

    python scripts/extract_isca_bucket.py --datadir $GFDL_DATA/bucket_grey_t21 \\
        --avg-months 12 --out baseline/reference/bucket_isca_t21.npz
"""
import argparse
from pathlib import Path

import netCDF4 as nc
import numpy as np

SECONDS_PER_DAY = 86400.0


def month_files(datadir: Path):
    """All atmos_monthly.nc, sorted by run index."""
    runs = sorted(datadir.glob("run*/atmos_monthly.nc"))
    if not runs:
        raise FileNotFoundError(f"no run*/atmos_monthly.nc under {datadir}")
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datadir", type=Path, required=True)
    ap.add_argument("--avg-months", type=int, default=12,
                    help="number of trailing months to average")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    files = month_files(args.datadir)
    window = files[-args.avg_months:]
    print(f"averaging {len(window)} months of {len(files)} available")

    bd = pr = ts = None
    lat = lon = None
    n = 0
    for f in window:
        d = nc.Dataset(f)
        if lat is None:
            lat = d.variables["lat"][:]
            lon = d.variables["lon"][:]
        # each monthly file holds one time-avg record; average across the window
        b = np.asarray(d.variables["bucket_depth"][:]).mean(axis=0)
        p = np.asarray(d.variables["precipitation"][:]).mean(axis=0) * SECONDS_PER_DAY
        t = np.asarray(d.variables["t_surf"][:]).mean(axis=0)
        bd = b if bd is None else bd + b
        pr = p if pr is None else pr + p
        ts = t if ts is None else ts + t
        n += 1
        d.close()
    bd, pr, ts = bd / n, pr / n, ts / n

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, lat=np.asarray(lat), lon=np.asarray(lon),
                        bucket_depth=bd, precip=pr, t_surf=ts, avg_months=n)
    print(f"wrote {args.out}  bucket_depth[{bd.min():.2f},{bd.max():.2f}] "
          f"precip global {pr.mean():.2f} mm/day  t_surf[{ts.min():.1f},{ts.max():.1f}]")


if __name__ == "__main__":
    main()
