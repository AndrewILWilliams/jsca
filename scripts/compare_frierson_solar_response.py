"""Does climate change look the same in jsca and Isca? +2% solar-constant response.

Loads control and +2%-solar ensemble members (8 monthly means each) for both
models at T21, forms each model's RESPONSE (perturbed mean − control mean), and
compares jsca's response to Isca's: global-mean surface warming and precip change,
the warming pattern T(lat,p), and per-field pattern correlation of the responses.
Member spread gives the ±1σ envelope on each global-mean response.

Run: ``python scripts/compare_frierson_solar_response.py``
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REF = "baseline/reference"


def gm(a, w):  # area-weighted mean over the last (lat) axis
    return (a * w).sum(-1) / w.sum()


def load(tag):
    c = np.load(f"{REF}/frierson_{tag}_members_t21.npz")
    p = np.load(f"{REF}/frierson_{tag}_members_t21_solar2pct.npz")
    return c, p


def main() -> None:
    argparse.ArgumentParser().parse_args()
    Ic, Ip = load("isca")
    Jc, Jp = load("jsca")
    lat, pfull = Ic["lat"], Ic["pfull"]
    w = np.cos(np.deg2rad(lat))

    def response(c, p, key):
        return p[f"{key}_members"].mean(0) - c[f"{key}_members"].mean(0)

    def gm_resp_pm(c, p, key, sc):  # global-mean response ± member sigma
        cm = gm(c[f"{key}_members"], w) * sc  # (8,)
        pm = gm(p[f"{key}_members"], w) * sc
        d = pm.mean() - cm.mean()
        se = np.sqrt(cm.var(ddof=1) + pm.var(ddof=1)) / np.sqrt(len(cm))
        return d, se

    print("=== +2% solar-constant response: jsca vs Isca (T21) ===\n")
    print("global-mean response       Isca              jsca")
    for key, sc, unit in [("t_surf", 1.0, "K"), ("precip", 86400.0, "mm/day")]:
        di, si = gm_resp_pm(Ic, Ip, key, sc)
        dj, sj = gm_resp_pm(Jc, Jp, key, sc)
        print(f"  d{key:7s} {di:+6.3f}±{si:.3f} {unit:7s}  {dj:+6.3f}±{sj:.3f} {unit}")
    # hydrological sensitivity %/K
    dTi, _ = gm_resp_pm(Ic, Ip, "t_surf", 1.0)
    dPi, _ = gm_resp_pm(Ic, Ip, "precip", 86400.0)
    dTj, _ = gm_resp_pm(Jc, Jp, "t_surf", 1.0)
    dPj, _ = gm_resp_pm(Jc, Jp, "precip", 86400.0)
    Pi0 = gm(Ic["precip_members"].mean(0), w) * 86400.0
    Pj0 = gm(Jc["precip_members"].mean(0), w) * 86400.0
    print(f"\nhydrological sensitivity: Isca {100 * dPi / Pi0 / dTi:+.1f} %/K   "
          f"jsca {100 * dPj / Pj0 / dTj:+.1f} %/K")

    print("\nresponse pattern correlation (jsca vs Isca):")
    for key in ["t_surf", "precip", "T", "u"]:
        ri, rj = response(Ic, Ip, key), response(Jc, Jp, key)
        corr = np.corrcoef(ri.ravel(), rj.ravel())[0, 1]
        print(f"  {key:7s} corr={corr:.3f}")

    # figure: T-response (Isca | jsca | diff), plus dt_surf(lat) and dprecip(lat)
    Ti, Tj = response(Ic, Ip, "T"), response(Jc, Jp, "T")
    fig = plt.figure(figsize=(16, 9))
    dl = np.linspace(-3, 3, 13)
    for i, (data, title) in enumerate([(Ti, "Isca ΔT"), (Tj, "jsca ΔT"),
                                       (Tj - Ti, "ΔT: jsca − Isca")]):
        a = fig.add_subplot(2, 3, i + 1)
        cf = a.contourf(lat, pfull, data, levels=dl, cmap="RdBu_r", extend="both")
        a.invert_yaxis()
        plt.colorbar(cf, ax=a, label="K")
        a.set_title(title)
        a.set_xlabel("latitude")
        a.set_ylabel("pressure (hPa)")
    a = fig.add_subplot(2, 3, 4)
    a.plot(lat, response(Ic, Ip, "t_surf"), "k-", label="Isca")
    a.plot(lat, response(Jc, Jp, "t_surf"), "r--", label="jsca")
    a.set_title("Δt_surf (K)")
    a.set_xlabel("latitude")
    a.legend()
    a.grid(alpha=0.3)
    a = fig.add_subplot(2, 3, 5)
    a.plot(lat, response(Ic, Ip, "precip") * 86400, "k-", label="Isca")
    a.plot(lat, response(Jc, Jp, "precip") * 86400, "r--", label="jsca")
    a.set_title("Δprecip (mm/day)")
    a.set_xlabel("latitude")
    a.legend()
    a.grid(alpha=0.3)
    a = fig.add_subplot(2, 3, 6)
    a.plot(lat, response(Ic, Ip, "u")[-1], "k-", label="Isca")
    a.plot(lat, response(Jc, Jp, "u")[-1], "r--", label="jsca")
    a.set_title("Δu near-surface (m/s)")
    a.set_xlabel("latitude")
    a.legend()
    a.grid(alpha=0.3)
    fig.suptitle("Frierson +2% solar-constant response: jsca vs Isca (T21)", fontsize=13)
    fig.tight_layout()
    fig.savefig("docs/figures/frierson_solar_response.png", dpi=120)
    print("\nsaved docs/figures/frierson_solar_response.png")


if __name__ == "__main__":
    main()
