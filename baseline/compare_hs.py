"""Compare jsca's Held-Suarez climatology against the pinned Isca reference.

Loads the committed Isca monthly members (``hs_isca_members.npz``, 8 x 30-day
means) and a jsca run (``hs_jsca_run.npz`` from ``bench/run_jsca_held_suarez.py``),
forms 8 matching jsca monthly members, and runs the Tier-3 equivalence test
(:func:`jsca.testing.ensemble_mean_test`): at each (level, latitude) point, is
jsca's mean within Isca's own month-to-month spread? FDR-controlled across all
points, with a practical-significance floor.

Also writes a 2x3 figure (Isca | jsca | difference, for u and T) to
``docs/figures/hs_jsca_vs_isca.png``.
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jsca.testing import ensemble_mean_test


def jsca_monthly_members(run: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    """8 monthly (30-day) zonal-mean members from the daily samples.

    ``u_daily`` is ``(ndays, nlat, nlon, K)`` (level k=0 top). Returns u, T as
    ``(8, K, nlat)`` to match Isca's ``(members, pfull, lat)``.
    """
    u = run["u_daily"].mean(axis=2)  # zonal mean -> (ndays, nlat, K)
    t = run["t_daily"].mean(axis=2)
    nd = u.shape[0]
    nm = nd // 30
    u = u[: nm * 30].reshape(nm, 30, u.shape[1], u.shape[2]).mean(1)  # (nm, nlat, K)
    t = t[: nm * 30].reshape(nm, 30, t.shape[1], t.shape[2]).mean(1)
    return np.moveaxis(u, -1, 1), np.moveaxis(t, -1, 1)  # (nm, K, nlat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--isca", default="baseline/reference/hs_isca_members.npz")
    ap.add_argument("--jsca", default="baseline/reference/hs_jsca_run.npz")
    ap.add_argument("--fig", default="docs/figures/hs_jsca_vs_isca.png")
    ap.add_argument("--u-floor", type=float, default=2.0, help="m/s")
    ap.add_argument("--t-floor", type=float, default=1.5, help="K")
    args = ap.parse_args()

    isca = np.load(args.isca)
    run = np.load(args.jsca)
    lat, pfull = isca["lat"], isca["pfull"]
    u_ctrl, t_ctrl = isca["u_members"], isca["t_members"]  # (8, K, nlat)
    u_test, t_test = jsca_monthly_members(run)
    print(f"members: Isca {u_ctrl.shape[0]}, jsca {u_test.shape[0]}")

    ru = ensemble_mean_test(u_ctrl, u_test, floor=args.u_floor)
    rt = ensemble_mean_test(t_ctrl, t_test, floor=args.t_floor)
    uc, ut = u_ctrl.mean(0), u_test.mean(0)
    tc, tt = t_ctrl.mean(0), t_test.mean(0)

    def stats(name, c, t, res, unit):
        d = t - c
        print(f"{name}: bias={d.mean():+.2f} rms={np.sqrt((d**2).mean()):.2f} {unit} | "
              f"max|diff|={np.abs(d).max():.2f} | "
              f"fail_fraction={res.fail_fraction:.1%} (floor-guarded, FDR 5%)")

    print(f"Isca jet: max u={uc.max():.1f} m/s ; jsca jet: max u={ut.max():.1f} m/s")
    stats("u", uc, ut, ru, "m/s")
    stats("T", tc, tt, rt, "K")

    # figure: rows = u, T ; cols = Isca, jsca, jsca-Isca
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
    ulev = np.arange(-30, 36, 4)
    dlev_u = np.linspace(-8, 8, 17)
    tlev = np.arange(190, 316, 10)
    dlev_t = np.linspace(-6, 6, 13)

    for j, (field, lev, dlev, cmap, unit, res) in enumerate([
        (("u", uc, ut), ulev, dlev_u, "RdBu_r", "m/s", ru),
        (("T", tc, tt), tlev, dlev_t, "viridis", "K", rt),
    ]):
        name, c, t = field
        for i, (data, title, lv, cm) in enumerate([
            (c, f"Isca {name}", lev, cmap),
            (t, f"jsca {name}", lev, cmap),
            (t - c, f"jsca - Isca {name}", dlev, "RdBu_r"),
        ]):
            a = ax[j, i]
            cf = a.contourf(lat, pfull, data, levels=lv, cmap=cm, extend="both")
            plt.colorbar(cf, ax=a, label=unit)
            if i == 2:  # stipple where the difference is judged significant
                yy, xx = np.where(res.reject)
                a.scatter(lat[xx], pfull[yy], s=2, c="k", alpha=0.5)
            a.set_title(title)
            if j == 1:
                a.set_xlabel("latitude")
            if i == 0:
                a.set_ylabel("pressure (hPa)")
    ax[0, 0].invert_yaxis()
    fig.suptitle("jsca vs Isca Held-Suarez climatology (T42L25; stipple = "
                 "significant difference, FDR 5% + floor)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=120)
    print(f"saved {args.fig}")


if __name__ == "__main__":
    main()
