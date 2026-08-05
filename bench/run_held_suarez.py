"""Spin up the Held-Suarez benchmark from rest and plot its climatology.

Runs the assembled dynamical core + HS forcing (:mod:`jsca.model.held_suarez`)
from an isothermal resting state, integrates to statistical equilibrium, and
time-/zonal-averages the zonal wind and temperature — the canonical HS94
diagnostic (eddy-driven midlatitude jets, tropopause temperature structure).

Usage:
    python bench/run_held_suarez.py --spinup-days 110 --sample-days 40 \
        --out docs/figures/pr14_held_suarez.png

At T21L15 with dt=600 s this is ~10 min on CPU. It reproduces the figure in the
Held-Suarez PR; there is no external input.
"""

from __future__ import annotations

import argparse

import jax.numpy as jnp
import numpy as np

import jsca  # noqa: F401
from jsca.grid.transforms import area_weighted_global_mean
from jsca.model import build_held_suarez, initial_state, integrate
from jsca.model.held_suarez import _grid_from_spectral


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-fourier", type=int, default=21)
    ap.add_argument("--num-levels", type=int, default=15)
    ap.add_argument("--dt", type=float, default=600.0)
    ap.add_argument("--spinup-days", type=int, default=110)
    ap.add_argument("--sample-days", type=int, default=40)
    ap.add_argument("--out", default="docs/figures/pr14_held_suarez.png")
    args = ap.parse_args()

    m = build_held_suarez(num_fourier=args.num_fourier, num_levels=args.num_levels, dt=args.dt)
    st = initial_state(m, temperature=264.0)
    tf = m.dyn.transforms
    lat = np.degrees(np.arcsin(np.asarray(tf.sin_lat)))
    bk = np.asarray(m.dyn.bk)
    sig = 0.5 * (bk[1:] + bk[:-1])
    spd = int(round(86400.0 / args.dt))

    st = integrate(m, st, args.spinup_days * spd)
    u, v, _t, _ps = _grid_from_spectral(m, *st, 1)
    ke = float(area_weighted_global_mean(tf, jnp.mean(0.5 * (u**2 + v**2), -1)))
    print(f"spun up {args.spinup_days} days: KE={ke:.1f}, max|u|={float(jnp.abs(u).max()):.1f} m/s")

    st, (u_s, t_s, _ps_s) = integrate(m, st, args.sample_days * spd, sample_every=spd)
    um = np.asarray(u_s).mean(axis=(0, 2))  # (nlat, K)
    tm = np.asarray(t_s).mean(axis=(0, 2))
    jmax = np.unravel_index(um.argmax(), um.shape)
    print(f"jet: max u={um.max():.1f} m/s at lat={lat[jmax[0]]:.0f}, sigma={sig[jmax[1]]:.2f}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    lev = np.linspace(-36, 36, 19)
    c = ax[0].contourf(lat, sig, um.T, levels=lev, cmap="RdBu_r", extend="both")
    ax[0].contour(lat, sig, um.T, levels=[0], colors="k", linewidths=0.6)
    plt.colorbar(c, ax=ax[0], label="m/s")
    ax[0].invert_yaxis()
    ax[0].set_ylabel("sigma")
    ax[0].set_xlabel("latitude")
    ax[0].set_title("zonal-mean zonal wind\n(eddy-driven midlatitude jets)")
    c2 = ax[1].contourf(lat, sig, tm.T, levels=15, cmap="viridis")
    plt.colorbar(c2, ax=ax[1], label="K")
    ax[1].invert_yaxis()
    ax[1].set_ylabel("sigma")
    ax[1].set_xlabel("latitude")
    ax[1].set_title("zonal-mean temperature")
    fig.suptitle(
        f"jsca Held-Suarez climatology (T{args.num_fourier}L{args.num_levels}, "
        f"{args.sample_days}-day mean)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
