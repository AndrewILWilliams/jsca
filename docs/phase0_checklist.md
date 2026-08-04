# Phase-0 closeout checklist

Ordered; items 1–3 need your cluster/node, the rest are done or code-only.

1. **Pin the baseline** — ~~identify the upstream Isca commit~~ done:
   `ExeClim/Isca@a290bc37` (master, 2026-01-30; see `baseline/PINNED.md`).
   Remaining: build on the target node via `baseline/native_build.sh` (or the
   shipped Dockerfile) and archive the build log + toolchain versions there.
2. **Measure Fortran throughput on the target node** —
   `python baseline/run_timing_case.py --case held_suarez --cores 1 4 8 16 --days 8`,
   then the same for `frierson`; extract per-step numbers from the FMS clock
   table with `baseline/parse_timings.py`. Commit `baseline/timings.json`.
   These numbers are the denominator of the Phase-1 performance gate (≥0.5×).
3. **Run and archive Control A** — `frierson` T42L25, 2 yr spin-up + 5 yr,
   N=5 members (perturbed initial T), daily means per scoping §4.5. Storage
   ~ a few GB. (Control B waits for Phase 4.)
4. ~~Grid/Legendre golden fixtures from real Fortran~~ — done in this kit
   (`tests/fixtures/grid_reference_t42.npz` via `dump_grid_reference.F90`);
   regenerate on the pinned build to confirm identical values.
5. **Extend the dump module into the model** — add `jsca_dump.F90` to
   `path_names`, instrument `two_stream_gray_rad` and `qe_moist_convection`
   first (per `fortran_instrumentation/README.md`), harvest from a spun-up
   day of Control A, commit `.npz` fixtures.
6. **Freeze Tier-3 thresholds** — encode the scoping §4.5 battery (fields,
   floors, alpha, thinning) as a config in `src/jsca/testing/` so the
   contract exists before any model comparison is possible.
7. **Repo home + CI** — push to Git hosting, enable the CI workflow, decide
   the real package name (jsca is a placeholder).
