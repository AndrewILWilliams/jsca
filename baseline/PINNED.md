# Pinned Fortran baseline — first build record (cloud sandbox, 2026-08-04)

The baseline was built and run from the `Iscamaster 2.zip` snapshot in an
Ubuntu 24.04 cloud sandbox as part of the Phase-0 starter kit. **Repeat this
on the target node and update this file** — these numbers establish the
recipe and a same-hardware JAX comparison, not your node's denominator.

## Toolchain that worked

| Component | Version |
|---|---|
| gfortran | 13.3.0 (Ubuntu 24.04) |
| OpenMPI | 4.1.6 |
| netCDF-Fortran | 4.5.4 |
| Codebase | `DryCodeBase` (`-DRRTM_NO_COMPILE -DSOC_NO_COMPILE`), `GFDL_ENV=docker` |
| **Snapshot pin** | **`ExeClim/Isca` commit `a290bc376d84d0ee83adbb80eb374b9f629c3534`** ("Merge pull request #292 from malcolmmaas/patch-1", 2026-01-30 10:49:02 UTC). Identified by exact archive-timestamp match (GitHub zips stamp entries with the committer time) and content-verified by spot-diff of constants.F90, gauss_and_legendre.F90, ReadMe.md, codebase.py against the upstream commit. All Tier-1 fixtures and baseline numbers refer to this commit. |

## Required deviations from the shipped recipe (apply on your node too)

1. **Modern-gfortran flags** — `src/extra/python/isca/templates/mkmf.template.gfort`
   (selected by `GFDL_ENV=docker`) needs `-fallow-invalid-boz
   -fallow-argument-mismatch` appended to `FFLAGS` (gfortran ≥ 10 treats
   legacy FMS constructs as errors; the `ubuntu_conda` template already has
   them, `gfort` does not).
2. **`pip install -e src/extra/python` fails** on Debian/Ubuntu-patched
   setuptools (legacy `setup.py develop` → `install_layout` AttributeError).
   Workaround: `export PYTHONPATH=$GFDL_BASE/src/extra/python`.
3. **The snapshot must be a git repo** — `isca.CodeBase.from_directory`
   requires it (`git init && git add -A && git commit` on the unzipped tree).
4. **Root MPI** (containers only): `OMPI_ALLOW_RUN_AS_ROOT=1` and
   `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`.

## Measured: Held–Suarez T42L25, dt=600 s, 8 model days (1152 steps)

FMS "Total runtime" (excludes python-side setup; run dirs archived in the
sandbox at `/tmp/gfdl_data/held_suarez_default/`; exact namelist in
`reference/input.nml`):

| Cores | Total runtime | Steps/s | Sim-years/day |
|---|---:|---:|---:|
| 1 | 109.28 s | 10.54 | 17.3 |
| 2 | 55.19 s | 20.87 | 34.3 |

Scaling 1→2 cores is essentially perfect (0.99×2).

## Same-hardware JAX comparison (the point of doing this here)

On the *same 2-core sandbox*, the jsca transform layer
(`bench/bench_transforms.py`, T42 L25 ×4 fields, float64) does **79
spectral↔grid roundtrips/s** — i.e. a budget of ~40 model steps/s if a step
costs ~2 roundtrip-equivalents, vs 20.9 steps/s for the *entire* Fortran
model. Read carefully: this says the JAX transform layer alone runs ~2× the
full Fortran step rate on identical hardware, which leaves real headroom for
the rest of the dycore — it does **not** yet demonstrate a full-step ≥0.5×.
That is precisely what the Phase-1 spike gate measures once the semi-implicit
step exists.

## For per-section Fortran clocks

Set in the experiment namelist: `fms_nml: clock_grain = 'ROUTINE'` (the
default coarse grain only reports "Total runtime"), then
`python parse_timings.py <run log> --steps N`.
