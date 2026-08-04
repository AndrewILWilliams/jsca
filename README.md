# jsca — Phase-0 starter kit for the Isca → JAX port

Companion to the scoping document in the *PortingISCA* project
(`scoping/isca-jax-scoping.md`). This kit is Phase 0 plus the first bricks of
Phase 1: the validated Gaussian grid + spectral-transform layer, the
statistical-equivalence toolkit, the Fortran fixture harvester, and the
baseline benchmarking scripts.

## What's here, and its status

| Piece | Status |
|---|---|
| `src/jsca/constants.py` | Ported verbatim from Isca's `constants.F90` (values are Isca's, incl. its non-textbook choices) |
| `src/jsca/grid/gaussian.py` | Faithful port of `compute_gaussian` (Newton/NR, hemisphere layout) + global S→N assembly |
| `src/jsca/grid/legendre.py` | Faithful port of `compute_legendre` — GFDL normalization (∫P̃² dμ = 1, no Condon–Shortley), GFDL `(m, n)` storage with `l = m + n` |
| `src/jsca/grid/spectral.py` | Truncation layout (T42/T85 presets), masks, Laplacian eigenvalues |
| `src/jsca/grid/transforms.py` | JAX transforms: rfft in λ, Legendre einsum in μ; jit/vmap-clean; documented conventions for later fixture matching |
| `src/jsca/dycore/` | **Time-stepping spine**: `leapfrog.py` (RAW-filtered leapfrog + two-level split, incl. the Fortran real-variant quirk), `press_and_geopot.py` (Simmons–Burridge/mcm pressures, hydrostatic geopotential), `matrix_invert.py` (semi-implicit solves; documented LAPACK deviation) — all fixture-validated against the real Fortran |
| `src/jsca/testing/equivalence.py` | Tier-3 tools: ensemble envelope test (t, df=N−1), BH-FDR, practical floors, KS |
| `tests/` | The suite described below — **run `pytest` first thing** |
| `fortran_instrumentation/` | `jsca_dump.F90` (fixture harvester), stubs + standalone driver that dumps **real Fortran** grid/Legendre reference data |
| `baseline/` | Native build script (mirrors Isca's Dockerfile), timing runner, FMS clock-table parser |
| `bench/` | Transform micro-benchmark (the first perf-gate row) |
| `docs/phase0_checklist.md` | What remains to close Phase 0, in order |

## Quick start

```bash
pip install -e .[dev]
pytest                      # grid/transform/stats suite (float64)
python bench/bench_transforms.py
```

Regenerating the Fortran golden fixtures (needs gfortran; `ISCA_SRC` = the
unzipped Isca snapshot):

```bash
cd fortran_instrumentation
gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
  fms_stubs.F90 jsca_dump.F90 \
  "$ISCA_SRC/src/atmos_spectral/tools/gauss_and_legendre.F90" \
  dump_grid_reference.F90 -o dump_grid_reference
mkdir -p ../tests/fixtures/raw
JSCA_DUMP_DIR=../tests/fixtures/raw ./dump_grid_reference
python -c "
from read_dumps import read_all
import numpy as np
d = {k: v[0] for k, v in read_all('../tests/fixtures/raw').items()}
np.savez_compressed('../tests/fixtures/grid_reference_t42.npz', **d)
"
pytest ../tests/test_fortran_fixtures.py -v
```

## Design rules (bind all future code)

1. Pure functions of `(config, params, state)`; no module-level mutable state.
2. Configs are frozen dataclasses (hashable, jit-static); numerical tables are
   float64 arrays precomputed at init into a `params` pytree.
3. Everything on the step path composes with `jit`/`vmap`/`scan`; no host
   syncs inside a chunk.
4. Faithful means faithful: port Isca's algorithm and constants, cite the
   Fortran source file in the docstring, and prove it with a fixture test.
   Don't "improve" numerics in the same commit as a port — ever.
5. Latitude ordering is south → north everywhere in jsca; reconcile with
   Fortran layouts at fixture boundaries only.

## Licence

GPL-3.0-or-later (derivative of Isca / GFDL FMS, both GPL-3.0).
