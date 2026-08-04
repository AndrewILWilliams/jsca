"""Transform micro-benchmark — the first row of the Phase-1 performance gate.

Times a T42L25 spectral->grid->spectral round trip (the dominant dycore cost
per step alongside the semi-implicit solve) under jit, single dispatch and
scanned, and reports a *transform-limited* upper bound on model throughput.

This is a proxy, not a model benchmark: real steps add physics, the implicit
solve, and several more transforms. Compare like-for-like against the
Fortran baseline's transform clocks (FMS clock 'Transforms' lines), not
against whole-model steps/s.

Run:  python bench/bench_transforms.py [--nlev 25] [--steps 200]
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import jsca  # noqa: F401  (x64)
from jsca.grid import T42_GRID, build_transforms, grid_to_spectral, spectral_to_grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlev", type=int, default=25)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--fields", type=int, default=4, help="fields per step (vor,div,T,q-ish)")
    args = ap.parse_args()

    params, _ = build_transforms(T42_GRID)
    rng = np.random.default_rng(0)
    spec = rng.standard_normal((args.fields, args.nlev, 43, 44)) * np.asarray(
        params.mask_prognostic
    )
    spec = jnp.asarray(spec + 0j)

    @jax.jit
    def roundtrip(s):
        return grid_to_spectral(params, spectral_to_grid(params, s))

    @jax.jit
    def scanned(s):
        def body(carry, _):
            return roundtrip(carry), None

        out, _ = jax.lax.scan(body, s, xs=None, length=args.steps)
        return out

    # warmup / compile
    roundtrip(spec).block_until_ready()
    t0 = time.perf_counter()
    scanned(spec).block_until_ready()
    t0 = time.perf_counter() - t0  # noqa: F841  (compile+run, not reported)

    t1 = time.perf_counter()
    scanned(spec).block_until_ready()
    dt = (time.perf_counter() - t1) / args.steps

    n_single = 50
    t2 = time.perf_counter()
    for _ in range(n_single):
        roundtrip(spec).block_until_ready()
    dt_single = (time.perf_counter() - t2) / n_single

    print(f"backend: {jax.default_backend()}, devices: {jax.devices()}")
    print(f"T42 L{args.nlev} x{args.fields} fields, float64")
    print(f"roundtrip (scanned): {dt*1e3:8.3f} ms   [{1.0/dt:8.1f} roundtrips/s]")
    print(f"roundtrip (single) : {dt_single*1e3:8.3f} ms")
    # transform-limited bound: assume ~2 roundtrip-equivalents per model step
    dt_step = 2 * dt
    dt_model = 600.0  # s, typical T42 timestep
    steps_per_sec = 1.0 / dt_step
    sypd = dt_model * steps_per_sec / 365.0  # (sim-seconds per wall-second) -> years/day
    print(
        f"transform-limited bound: {steps_per_sec:,.0f} steps/s, "
        f"~{sypd:,.0f} sim-years/day IF transforms were the only cost"
    )


if __name__ == "__main__":
    main()
