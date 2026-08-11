"""Bucket-hydrology climatology: jsca vs the real Isca run (T21, grey radiation).

Loads the jsca and Isca time-mean field ``.npz`` files
(``scripts/run_jsca_bucket_climatology.py`` and ``scripts/extract_isca_bucket.py``)
and compares the two land-relevant fields on the shared T21 grid:

* ``bucket_depth`` [m] — the soil-moisture reservoir (land only);
* ``precip`` [mm/day];
* ``t_surf`` [K].

Prints skill statistics (land-mean, bias, RMSE, pattern correlation over land for
the bucket) and writes a maps figure (jsca | Isca | difference) per field.

Both grids are the T21 32x64 Gaussian grid; Isca stores latitude the same
south->north order as jsca, but the loader checks and flips if needed.

Usage::

    python scripts/compare_bucket_climatology.py \\
        --jsca baseline/reference/bucket_jsca_t21.npz \\
        --isca baseline/reference/bucket_isca_t21.npz \\
        --out baseline/reference/bucket_climatology_t21.png
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _orient(iscad, jlat):
    """Return Isca fields oriented to match jsca's south->north latitude."""
    ilat = iscad["lat"]
    flip = np.sign(ilat[1] - ilat[0]) != np.sign(jlat[1] - jlat[0])
    out = {}
    for k in ("bucket_depth", "precip", "t_surf"):
        arr = iscad[k]
        out[k] = arr[::-1, :] if flip else arr
    return out


def _stats(name, j, i, mask=None):
    if mask is not None:
        j, i = j[mask], i[mask]
    bias = float((j - i).mean())
    rmse = float(np.sqrt(((j - i) ** 2).mean()))
    ja, ia = j - j.mean(), i - i.mean()
    denom = np.sqrt((ja ** 2).sum() * (ia ** 2).sum())
    corr = float((ja * ia).sum() / denom) if denom > 0 else float("nan")
    print(f"  {name:14s} jsca_mean={j.mean():8.3f} isca_mean={i.mean():8.3f} "
          f"bias={bias:+8.3f} rmse={rmse:7.3f} corr={corr:5.2f}")
    return bias, rmse, corr


def _panel(ax, lat, lon, field, title, cmap, vlim=None, land=None):
    kw = {} if vlim is None else {"vmin": vlim[0], "vmax": vlim[1]}
    m = ax.pcolormesh(lon, lat, field, cmap=cmap, shading="auto", **kw)
    if land is not None:
        ax.contour(lon, lat, land, levels=[0.5], colors="k", linewidths=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    plt.colorbar(m, ax=ax, shrink=0.8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsca", type=Path, required=True)
    ap.add_argument("--isca", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    J = np.load(args.jsca)
    I = np.load(args.isca)
    lat, lon = J["lat"], J["lon"]
    land = J["land"]
    landm = land > 0.5
    Io = _orient(I, lat)

    print("Bucket climatology skill (jsca vs Isca, T21 grey radiation):")
    _stats("bucket(land)", J["bucket_depth"], Io["bucket_depth"], landm)
    _stats("precip", J["precip"], Io["precip"])
    _stats("t_surf", J["t_surf"], Io["t_surf"])
    _stats("t_surf(land)", J["t_surf"], Io["t_surf"], landm)

    fields = [
        ("bucket_depth", "bucket depth [m]", "YlGnBu", (0, 2)),
        ("precip", "precip [mm/day]", "viridis", None),
        ("t_surf", "t_surf [K]", "RdBu_r", None),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), constrained_layout=True)
    for row, (key, label, cmap, vlim) in enumerate(fields):
        jf, if_ = J[key], Io[key]
        if vlim is None:
            vlim = (min(jf.min(), if_.min()), max(jf.max(), if_.max()))
        _panel(axes[row, 0], lat, lon, jf, f"jsca {label}", cmap, vlim, land)
        _panel(axes[row, 1], lat, lon, if_, f"Isca {label}", cmap, vlim, land)
        dlim = np.abs(jf - if_).max()
        _panel(axes[row, 2], lat, lon, jf - if_, f"jsca - Isca {label}",
               "RdBu_r", (-dlim, dlim), land)
    fig.suptitle("Bucket hydrology T21, grey radiation — jsca vs Isca", fontsize=12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
