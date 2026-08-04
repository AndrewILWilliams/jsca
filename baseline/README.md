# Baseline: pinning and measuring the Fortran Isca

Phase-0 requires a frozen, measured Fortran reference (scoping doc §4.2,
§3.4). Workflow:

1. `ISCA_SRC=/path/to/Iscamaster ./native_build.sh` — system deps (mirrors
   Isca's own Dockerfile), python front end, env vars. Read
   **PINNED.md → "Required deviations"** first; all four gotchas found during
   the first build are listed there with fixes.
2. `python run_timing_case.py --case held_suarez --cores 1 4 8 16 --days 8`
   then `--case frierson`. Appends to `timings.json`.
3. Per-step numbers: `python parse_timings.py <logfile> --steps N`
   (with `fms_nml: clock_grain='ROUTINE'` for section-level clocks).
4. Record everything in `PINNED.md`; archive run directories ($GFDL_DATA).
5. Control A (the 5-year equivalence ensemble) is checklist item 3 in
   `docs/phase0_checklist.md` — start it once timings look sane.

`reference/` holds the exact `input.nml` and snapshot git hash of the first
pinned run (sandbox build, 2026-08-04).
