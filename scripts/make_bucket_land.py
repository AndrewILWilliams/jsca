"""Generate the T21 realistic-continents ``land.nc`` for the bucket experiment.

Writes a ``land.nc`` in Isca's ``land_option='input'`` format (float32 ``lat``,
``lon``, ``zsurf``, ``land_mask`` on the T21 32x64 Gaussian grid) from
:func:`jsca.model.land.continents_land_mask`. The SAME file is read by both the
Isca run and the jsca ``bucket_model`` so the land geography is identical
(continents, not Isca's shipped square block — see ``docs/bucket_model.md``).

The grid is jsca's T21 Gaussian latitudes (south -> north; they match Isca's own
T21 Gaussian latitudes to float precision) and evenly spaced longitudes. Surface
geopotential is zero (``zsurf = 0``), matching the flat-surface bucket test case.

Usage::

    python scripts/make_bucket_land.py --nlat 32 --nlon 64 --out /path/to/land.nc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from jsca.grid.gaussian import gaussian_grid
from jsca.model.land import continents_land_mask


def build_land(nlat: int, nlon: int):
    """Return ``(lat_deg, lon_deg, land_mask)`` on the T``(nlat/2-1)`` grid."""
    lat = np.degrees(gaussian_grid(nlat).lat)
    lon = np.arange(nlon) * 360.0 / nlon
    land = continents_land_mask(lat, lon)
    return lat, lon, land


def write_land_nc(path: Path, lat, lon, land):
    ds = Dataset(path, "w", format="NETCDF3_CLASSIC")
    ds.createDimension("lat", lat.size)
    ds.createDimension("lon", lon.size)
    vlat = ds.createVariable("lat", "f4", ("lat",))
    vlon = ds.createVariable("lon", "f4", ("lon",))
    vz = ds.createVariable("zsurf", "f4", ("lat", "lon"))
    vland = ds.createVariable("land_mask", "f4", ("lat", "lon"))
    vlat[:] = lat.astype(np.float32)
    vlon[:] = lon.astype(np.float32)
    vz[:] = np.zeros((lat.size, lon.size), dtype=np.float32)
    vland[:] = land.astype(np.float32)
    ds.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nlat", type=int, default=32)
    ap.add_argument("--nlon", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    lat, lon, land = build_land(args.nlat, args.nlon)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_land_nc(args.out, lat, lon, land)
    print(f"wrote {args.out}  ({args.nlat}x{args.nlon}, land fraction {land.mean():.4f})")


if __name__ == "__main__":
    main()
