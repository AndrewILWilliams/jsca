"""Run jsca's Held-Suarez in Isca's exact benchmark configuration.

Matches the pinned Isca reference (``baseline/reference/HS_REFERENCE.md``):
**T42 on the 64x128 Gaussian grid, 25 uneven-sigma levels**
(``scale_heights=6``, ``surf_res=0.5``, ``exponent=7.5``), ``damping_order=4``,
``dt=600 s``, ``reference_sea_level_press=1e5``. Integrated from a resting
isothermal state.

Output (``--out``, an .npz):
  - ``u_daily, t_daily, ps_daily`` — daily grid samples over the averaging
    window, ``(ndays, nlat, nlon, K)`` / ``(ndays, nlat, nlon)`` (level k=0 top).
  - ``lat`` (deg, S->N), ``pfull`` (hPa, from the reference so both sit on one
    axis), ``sigma_full``.

Feed to ``baseline/compare_hs.py`` for the jsca-vs-Isca climatology comparison.

Cost: ~1.5 h on CPU for the default 120-day spin-up + 240-day sample.
"""
from __future__ import annotations

import argparse
import time

import jax.numpy as jnp
import numpy as np

import jsca  # noqa: F401
from jsca.grid.transforms import area_weighted_global_mean
from jsca.model import build_held_suarez, initial_state, integrate
from jsca.model.held_suarez import _grid_from_spectral


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spinup-days", type=int, default=120)
    ap.add_argument("--sample-days", type=int, default=240)
    ap.add_argument("--dt", type=float, default=600.0)
    ap.add_argument("--ref", default="baseline/reference/hs_isca_reference.npz")
    ap.add_argument("--out", default="baseline/reference/hs_jsca_run.npz")
    args = ap.parse_args()

    ref = np.load(args.ref)
    spd = int(round(86400.0 / args.dt))  # steps per day

    # Isca's exact T42L25 uneven-sigma HS config, on Isca's 64x128 grid.
    m = build_held_suarez(
        num_fourier=42, nlat=64, nlon=128, num_levels=25, dt=args.dt,
        vert_coord_option="uneven_sigma", scale_heights=6.0, surf_res=0.5,
        exponent=7.5, damping_order=4, reference_sea_level_press=1.0e5,
    )
    tf = m.dyn.transforms
    lat = np.degrees(np.arcsin(np.asarray(tf.sin_lat)))
    bk = np.asarray(m.dyn.bk)
    sigma_full = 0.5 * (bk[1:] + bk[:-1])
    assert np.allclose(lat, ref["lat"], atol=1e-3), "grid mismatch vs reference"

    st = initial_state(m, temperature=264.0)

    # Spin up in 10-day chunks with flushed progress + a blow-up guard, so a long
    # background run reports early and aborts on instability instead of grinding
    # to a NaN. (T42L25 is a stiffer stability test than the validated T21L15.)
    t0 = time.time()
    chunk = 10
    for d0 in range(0, args.spinup_days, chunk):
        st = integrate(m, st, chunk * spd)
        u, v, _t, _ps = _grid_from_spectral(m, *st, 1)
        ke = float(area_weighted_global_mean(tf, jnp.mean(0.5 * (u**2 + v**2), -1)))
        umax, isnan = float(jnp.abs(u).max()), bool(jnp.isnan(u).any())
        print(f"  day {d0 + chunk:3d}: KE={ke:7.1f}  max|u|={umax:5.1f} m/s  "
              f"nan={isnan}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if isnan or umax > 200.0:
            raise SystemExit(f"unstable at day {d0 + chunk} (max|u|={umax}, nan={isnan})")
    print(f"spun up {args.spinup_days} d in {(time.time() - t0) / 60:.1f} min", flush=True)

    # Sample in 30-day chunks and reduce each to a monthly zonal-mean member
    # immediately. Reducing inside the loop (not stacking all daily 4-D fields in
    # one scan) keeps peak memory ~one month of daily samples instead of the whole
    # window -- the full-window stack OOM-kills the process at T42L25.
    nmonths = args.sample_days // 30
    u_members, t_members = [], []  # each (K, nlat), zonal+time mean
    ke_daily = []  # area-weighted global mean of 0.5<u^2> per day (a scalar series)
    for mo in range(nmonths):
        st, (u_s, t_s, _ps_s) = integrate(m, st, 30 * spd, sample_every=spd)
        u_s = np.asarray(u_s)  # (30, nlat, nlon, K)
        t_s = np.asarray(t_s)
        u_members.append(np.moveaxis(u_s.mean(axis=(0, 2)), -1, 0))  # (K, nlat)
        t_members.append(np.moveaxis(t_s.mean(axis=(0, 2)), -1, 0))
        ke_daily.extend(
            float(area_weighted_global_mean(tf, jnp.mean(0.5 * u_s[d] ** 2, -1)))
            for d in range(u_s.shape[0])
        )
        print(f"  month {mo + 1}/{nmonths} sampled "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    np.savez_compressed(
        args.out,
        u_members=np.array(u_members), t_members=np.array(t_members),  # (nmonths, K, nlat)
        ke_daily=np.array(ke_daily),
        lat=lat, pfull=ref["pfull"], sigma_full=sigma_full,
        spinup_days=args.spinup_days, sample_days=args.sample_days, dt=args.dt,
    )
    print(f"saved {args.out} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
