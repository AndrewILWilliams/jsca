"""Tier-3 statistical parity: jsca vs Isca Frierson climatology (the #27 closer).

Loads monthly (30-day) ensemble members — 8 per model, from the equilibrated
window of a T21 run — and runs :func:`jsca.testing.ensemble_mean_test` at each
(level, latitude) point: is jsca's ensemble mean within Isca's own month-to-month
internal variability? FDR-controlled across all points, with a practical-
significance floor per field. A low ``fail_fraction`` across the battery is the
within-sampling-parity verdict.

Members produced by ``scratchpad/jsca_t21_members.py`` (jsca) and extracted from
the Isca ``atmos_daily.nc`` (control). Run:
``python scripts/compare_frierson_ensemble.py --isca <isca.npz> --jsca <jsca.npz>``
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jsca.testing import ensemble_mean_test  # noqa: E402

# field: (scale, unit, practical floor in scaled units)
FIELDS = {
    "u": (1.0, "m/s", 2.0),
    "T": (1.0, "K", 1.5),
    "sphum": (1e3, "g/kg", 0.5),
    "t_surf": (1.0, "K", 1.5),
    "precip": (86400.0, "mm/day", 0.5),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--isca", default="/tmp/isca_t21_members_ctrl.npz")
    ap.add_argument("--jsca", default="/tmp/jsca_t21_members_ctrl.npz")
    ap.add_argument("--fig", default="docs/figures/frierson_ensemble_parity.png")
    args = ap.parse_args()

    I, J = np.load(args.isca), np.load(args.jsca)
    lat, pfull = I["lat"], I["pfull"]
    n = min(I["u_members"].shape[0], J["u_members"].shape[0])
    print(f"members: Isca {I['u_members'].shape[0]}, jsca {J['u_members'].shape[0]} "
          f"(using {n}); window = 8 x 30-day months, equilibrated")

    results = {}
    print("\nfield    bias      rms     max|d|   fail_fraction (FDR 5% + floor)")
    for f, (sc, unit, floor) in FIELDS.items():
        c = I[f"{f}_members"][:n] * sc
        t = J[f"{f}_members"][:n] * sc
        r = ensemble_mean_test(c, t, floor=floor)
        results[f] = (c.mean(0), t.mean(0), r)
        d = t.mean(0) - c.mean(0)
        print(f"{f:7s} {d.mean():+7.3f} {np.sqrt((d**2).mean()):7.3f} "
              f"{np.abs(d).max():7.3f} {unit:7s} {r.fail_fraction:6.1%}")

    verdict = max(r.fail_fraction for _, _, r in results.values())
    print(f"\nWORST fail_fraction across fields: {verdict:.1%} "
          f"-> {'PARITY (within sampling)' if verdict < 0.05 else 'residual differences'}")

    # figure: 2D fields u, T (Isca | jsca | diff w/ significance stipple)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
    for row, (f, dl, cmap, lv) in enumerate([
        ("u", np.linspace(-6, 6, 13), "RdBu_r", np.arange(-30, 40, 4)),
        ("T", np.linspace(-3, 3, 13), "viridis", np.arange(190, 306, 10)),
    ]):
        c, t, r = results[f]
        unit = FIELDS[f][1]
        for col, (data, title, levs, cm) in enumerate([
            (c, f"Isca {f}", lv, cmap), (t, f"jsca {f}", lv, cmap),
            (t - c, f"jsca - Isca {f}", dl, "RdBu_r")]):
            a = ax[row, col]
            cf = a.contourf(lat, pfull, data, levels=levs, cmap=cm, extend="both")
            plt.colorbar(cf, ax=a, label=unit)
            if col == 2:
                yy, xx = np.where(r.reject)
                a.scatter(lat[xx], pfull[yy], s=3, c="k", alpha=0.6)
            a.set_title(title)
            if row == 1:
                a.set_xlabel("latitude")
            if col == 0:
                a.set_ylabel("pressure (hPa)")
    ax[0, 0].invert_yaxis()
    fig.suptitle("Frierson T21 statistical parity: jsca vs Isca (8x30-day members; "
                 "stipple = significant diff, FDR 5% + floor)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=120)
    print(f"saved {args.fig}")


if __name__ == "__main__":
    main()
