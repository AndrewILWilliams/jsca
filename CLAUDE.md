# CLAUDE.md

## Think Before Coding
This is a complex codebase with many interdependencies and intricate scientific formulations. Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

 - State your assumptions explicitly. If uncertain, ask.
 - If multiple interpretations exist, present them - don't pick silently.
 - If a simpler approach exists, say so. Push back when warranted.
 - If something is unclear, stop. Name what's confusing. Ask.

Always document these decisions in the comments, and if appropriate in the documentation (and possibly in the high-level design documentation)

Comments should always reference the current state of the code, and explain *why* it is doing what it is doing, not how it is different to some previous version of the code (Which can get out of date and confusing)

# CLAUDE.md — standing instructions for Claude sessions on jsca

jsca is a **faithful JAX port of the Isca idealized GCM** (Fortran, GFDL
spectral core). Goal: statistically reproduce a pinned Fortran control run
(scoping doc: `docs/scoping.md`; working checklist: `docs/phase0_checklist.md`;
baseline record: `baseline/PINNED.md`). Fidelity is the product — a fast
model that doesn't match Isca is a failure.

## Iron rules

1. **Faithful means faithful.** Port Isca's algorithm, constants, and call
   order. Cite the Fortran source file (and lines for subtleties) in the
   docstring. Never "fix" or modernize Fortran behavior — reproduce it and
   flag oddities in the docstring (example: the real-variant
   `leapfrog_2level_A` RAW quirk, `jsca/dycore/leapfrog.py`). Any deliberate
   deviation (e.g. LAPACK in `matrix_invert.py`) must be documented in the
   docstring with its fixture-proven tolerance.
2. **Every ported routine ships with Tier-1 fixtures from the real Fortran**
   (workflow below). Tolerances: pure arithmetic rtol ≤ 1e-14; log/exp-bearing
   ≤ 1e-13; documented algorithm deviations ≤ 1e-11. Tests live in
   `tests/test_*_fixtures.py`; follow `tests/test_dycore_fixtures.py` as the
   template.
3. **Fixtures come only from Fortran.** Never generate, patch, or "refresh" a
   fixture `.npz` from Python output. If you cannot run gfortran in your
   environment, port the code, test against *committed* fixtures where they
   exist, and leave a clearly marked TODO + driver code for fixture
   generation — do not fake golden data.
4. **JAX discipline:** float64 (x64 is enabled by `import jsca`); pure
   functions of `(config, params, state)`; no module-level state; everything
   on the step path jit/vmap/scan-safe; configs static/hashable, numeric
   tables precomputed into params. No NumPy in hot paths (init/tests fine).
5. **Conventions:** latitude south→north; spectral storage `(m, n)` with
   total wavenumber `l = m + n` (see `jsca/grid/spectral.py`); transforms act
   on `(..., nlat, nlon)`; column physics put the level axis **last**
   (k = 0 top … K surface). Reconcile layouts only at fixture boundaries.
6. **Git:** commit as `Claude <noreply@anthropic.com>` — never the user's
   email. Small commits, tests included. `ruff check src tests bench` and
   `pytest` must pass before any push/PR. Work on a branch and open a PR
   unless told otherwise.
7. **Intelligible PRs:** For every PR, include a "Plain Language Summary" to explain what is going on to me (I am an atmospheric physicist, but not an accomplished programmer). Potentially include plots in the PRs, if relevant and helpful.

## The pinned Fortran reference

```bash
git clone https://github.com/ExeClim/Isca /tmp/isca && cd /tmp/isca
git checkout a290bc376d84d0ee83adbb80eb374b9f629c3534   # the validation target
export ISCA_SRC=/tmp/isca
```

All ports and fixtures refer to this commit. Do not port from a different
Isca version.

## Fixture workflow (per module)

1. Read the Fortran fully (`$ISCA_SRC/src/...`). Note `use` dependencies.
2. If deps are only `fms_mod`/`constants_mod` (+ small stubs), compile it
   unmodified with the stubs in `fortran_instrumentation/fms_stubs.F90`
   (extend stubs as needed — stub values must mirror Isca's `constants.F90`
   and `jsca/constants.py` exactly; stubs may contain logging only, never
   reimplemented numerics — if a stub needs numerics, dump the real values
   instead or stub the module by replicating the source file it wraps).
3. Write a standalone driver like
   `fortran_instrumentation/dump_dycore_reference.F90` (build line in its
   header): random-but-physical inputs, dump inputs and outputs via
   `jsca_dump_mod` (`JSCA_DUMP_DIR=...`), convert with `read_dumps.py` to a
   compressed `.npz` under `tests/fixtures/`.
4. Compiler: `gfortran -O2 -fdefault-real-8 -fdefault-double-8
   -ffree-line-length-none` (add `-fallow-invalid-boz
   -fallow-argument-mismatch` for FMS-heavy sources). `apt-get install
   gfortran` if missing.
5. Port to Python, write the fixture test, hit the rule-2 tolerances.
   Fortran index args in fixtures are 1-based; convert in tests.

## State of the port

Done and fixture-validated: `jsca.grid` (Gaussian grid, GFDL-normalized
associated Legendre, spectral transforms, Laplacian/hyperdiffusion),
`jsca.dycore` (leapfrog/RAW incl. two-level split, matrix_invert,
press_and_geopot), `jsca.testing` (Tier-3 equivalence stats), constants.

## Next queue (in order; one item per session is a good size)

1. **`spectral_damping.F90`** — deps: fms + `transforms_mod`
   (`get_eigen_laplacian`, `get_spec_domain`). For the fixture driver, stub
   `transforms_mod`: `get_spec_domain` returns the full serial domain;
   `get_eigen_laplacian` must reproduce Isca's eigenvalues — read how
   `$ISCA_SRC/src/atmos_spectral/tools/transforms.F90` /
   `spherical_fourier.F90` build them (sign and radius conventions!) and have
   the stub carry values computed by that same formula; cross-check against
   `jsca.grid.spectral.laplacian_eigenvalues` in the test.
2. **`implicit.F90`** — the semi-implicit heart. Read fully first; its
   dependencies (press_and_geopot, matrix_invert) are already ported.
3. **Spherical operators** (`spherical.F90`: u,v ↔ vor,div, gradients,
   ∂/∂μ using the l = M+1 derivative row) — prerequisite for
   spectral_dynamics; extends `jsca/grid/transforms.py`.
4. **`fv_advection.F90`**, `water_borrowing.F90`, `global_integral.F90`.
5. **`spectral_dynamics.F90` assembly** + Held–Suarez run and the
   climatology + ≥0.5×-Fortran performance gate (Fortran numbers:
   `baseline/reference/timings_sandbox.json`; node numbers pending).

## Climatology validation resolution

**Always validate a climatology-affecting change at T21 first, before any higher
resolution.** A 1-year T21 run (32×64) is ~35 min; the equivalent T42 (64×128) is
~2.5 h. Both jsca and the pinned Isca `frierson_test_case` run cheaply at T21, so
the jsca-vs-Isca comparison (same IC, same window; `run_frierson_climatology.py`,
`extract_isca_evolution.py`, `plot_frierson_evolution.py`,
`compare_frierson_climatology.py`) should be confirmed at T21 and only then re-run
at T42 to refresh the headline. This is how the water-conservation and
`damping_order` fixes were validated (`docs/frierson_climatology.md`).

## Performance

`bench/bench_transforms.py` is the pattern: jit + `lax.scan`, report per-step
cost honestly, compare like-for-like against the Fortran clocks. Never sync
host-side inside a scanned loop. Performance work must not change numerics —
fixtures gate that.
