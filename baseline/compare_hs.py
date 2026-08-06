"""Compare jsca's Held-Suarez climatology against the pinned Isca reference.

Loads the committed Isca monthly members (``hs_isca_members.npz``, 8 x 30-day
means) and the matching jsca monthly members (``hs_jsca_run.npz`` from
``bench/run_jsca_held_suarez.py``), and runs the Tier-3 equivalence test
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
from scipy import stats

from jsca.testing import ensemble_mean_test


def stats_mod_ks(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    res = stats.ks_2samp(np.ravel(a), np.ravel(b))
    return float(res.statistic), float(res.pvalue)


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
    u_test, t_test = run["u_members"], run["t_members"]  # (nmonths, K, nlat)
    n = min(u_ctrl.shape[0], u_test.shape[0])  # equal ensemble size for the test
    u_ctrl, t_ctrl, u_test, t_test = u_ctrl[:n], t_ctrl[:n], u_test[:n], t_test[:n]
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
    # distributional index: KS on the pooled zonal-mean values across all members
    ks_u = stats_mod_ks(u_ctrl, u_test)
    ks_t = stats_mod_ks(t_ctrl, t_test)
    print(f"KS (pooled zonal-mean distribution): u D={ks_u[0]:.3f} p={ks_u[1]:.2f} ; "
          f"T D={ks_t[0]:.3f} p={ks_t[1]:.2f}")

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
