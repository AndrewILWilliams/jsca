"""Run a jsca Frierson climatology and write the reference `.npz` files.

Produces, for a T42 run on Isca's exact 64x128 grid with Isca's matched initial
condition (``jsca.model.frierson.initial_state`` defaults):

  * ``frierson_jsca_zonalmean_t42x64.npz`` -- zonal-mean climatology over the
    averaging window (the like-for-like comparison target);
  * ``frierson_jsca_evolution_t42x64.npz`` -- the global-mean T / precip / t_surf
    time series (spin-up + averaging window), for the evolution plot.

The averaging-window chunk-mean scalar accumulator is reset each chunk; the
spatial climatology accumulates over the whole window. Global means are
area-weighted (cos-lat); temperature is additionally mass-weighted over the
sigma layers.

Usage: ``python scripts/run_frierson_climatology.py [spinup_days] [avg_days] [out_dir]``
Defaults: 200-day spin-up, 100-day average, writing to ``baseline/reference``.
A 300-day T42 run is ~1.8 h on a single CPU core. See ``docs/frierson_climatology.md``.
"""
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jsca  # noqa: F401  (enables float64)
from jsca.model.frierson import (
    FRIERSON_BK,
    _grid_from_spectral,
    _step_full,
    build_frierson,
    initial_state,
)

DT = 720.0
NLAT = 64
NLON = 128
CHUNK = 200
spinup_days = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0
avg_days = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else \
    Path(__file__).resolve().parent.parent / "baseline" / "reference"
SPINUP = int(round(spinup_days * 86400 / DT))
AVG = int(round(avg_days * 86400 / DT))

m = build_frierson(num_fourier=42, nlat=NLAT, nlon=NLON, dt=DT)
s = initial_state(m)   # Isca-matched IC
lat = np.arcsin(np.asarray(m.dyn.transforms.sin_lat))
wlat = jnp.asarray(np.cos(lat))
Wsum = float(np.cos(lat).sum())
dbk = jnp.asarray(FRIERSON_BK[1:] - FRIERSON_BK[:-1])   # sigma-layer thickness, sums to 1


def gm2d(f):
    return jnp.sum(wlat[:, None] * f) / (Wsum * NLON)


def scalars(s2, precip):
    _, _, t, _ = _grid_from_spectral(m, s2[0], s2[1], s2[2], s2[3], 1)
    return jnp.array([gm2d(jnp.sum(t * dbk, axis=-1)), gm2d(precip) * 86400.0, gm2d(s2[5])])


def diag_fields(s2):
    u, v, t, ps = _grid_from_spectral(m, s2[0], s2[1], s2[2], s2[3], 1)
    return {"ucomp": u, "vcomp": v, "temp": t, "sphum": s2[4][..., 1], "ps": ps, "t_surf": s2[5]}


def body_ts(carry, _):
    s, acc = carry
    s2, p = _step_full(m, s)
    return (s2, acc + scalars(s2, p)), None


def body_avg(carry, _):
    s, acc, facc = carry
    s2, p = _step_full(m, s)
    d = diag_fields(s2)
    d["precip"] = p
    return (s2, acc + scalars(s2, p), {k: facc[k] + d[k] for k in facc}), None


scan_spin = jax.jit(lambda s: jax.lax.scan(body_ts, (s, jnp.zeros(3)), None, length=CHUNK))
scan_avg = jax.jit(lambda c: jax.lax.scan(body_avg, c, None, length=CHUNK))
days, tsT, tsP, tsTs = [], [], [], []


def record(nstep, sums):
    mean = np.asarray(sums) / CHUNK
    days.append(nstep * DT / 86400.0)
    tsT.append(float(mean[0]))
    tsP.append(float(mean[1]))
    tsTs.append(float(mean[2]))


t0 = time.time()
s = jax.jit(lambda st: _step_full(m, st, m.dt, m.wave_matrix_cold)[0])(s)
s = jax.block_until_ready(s)
nstep = 1
while nstep < SPINUP:
    (s, sums), _ = scan_spin(s)
    s = jax.block_until_ready(s)
    nstep += CHUNK
    record(nstep, sums)
    print("spinup %5d day%6.1f gm_T=%.2f gm_precip=%.3f gm_tsurf=%.2f  %.1fms/step" % (
        nstep, days[-1], tsT[-1], tsP[-1], tsTs[-1], (time.time() - t0) / nstep * 1e3), flush=True)

facc0 = {**{k: jnp.zeros_like(v) for k, v in diag_fields(s).items()},
         "precip": jnp.zeros(m.lat2d.shape)}
carry = (s, jnp.zeros(3), facc0)
navg = 0
while navg < AVG:
    carry, _ = scan_avg((carry[0], jnp.zeros(3), carry[2]))   # reset per-chunk scalar acc
    carry = jax.block_until_ready(carry)
    navg += CHUNK
    nstep += CHUNK
    record(nstep, carry[1])
    print("avg %5d day%6.1f gm_T=%.2f gm_precip=%.3f gm_tsurf=%.2f  %.1fs" % (
        navg, days[-1], tsT[-1], tsP[-1], tsTs[-1], time.time() - t0), flush=True)

state, _, facc = carry
perf = (time.time() - t0) / (SPINUP + AVG) * 1e3
zm = {k: np.asarray(v).mean(axis=1) / AVG for k, v in facc.items()}
zm["lat_jsca"] = np.degrees(lat)
zm["_perf_ms_per_step"] = np.array(perf)
zm["meta"] = np.array([int(spinup_days + avg_days), int(spinup_days)])
np.savez_compressed(out_dir / "frierson_jsca_zonalmean_t42x64.npz", **zm)
np.savez(out_dir / "frierson_jsca_evolution_t42x64.npz", day=np.array(days), gm_T=np.array(tsT),
         gm_precip=np.array(tsP), gm_tsurf=np.array(tsTs), spinup_end_day=np.array(spinup_days))
print("DONE %.1f min, %.1f ms/step (64x128); wrote references to %s" % (
    (time.time() - t0) / 60, perf, out_dir), flush=True)
