# Overview & design

## What jsca is

`jsca` is a Python/JAX reimplementation of [Isca](https://execlim.github.io/Isca/),
the University of Exeter idealized global climate model built on the GFDL
spectral dynamical core. It ports the **dry dynamical core** and the
**grey-radiation moist-aquaplanet physics suite** — the configurations most used
for idealized climate dynamics — faithfully enough to statistically reproduce a
pinned Isca control run.

Isca itself is roughly 343k lines of Fortran/C, but only a small fraction is
the science of the configurations `jsca` targets. Most of the rest is GFDL FMS
infrastructure (MPI, I/O, diagnostics, time management) whose correct fate in a
Python rewrite is *replacement* — by JAX, xarray, and cftime — not porting. The
true port surface is roughly 60k lines of Fortran, which becomes an estimated
15–25k lines of Python.

## Why port it at all

Bit-for-bit reproduction of a Fortran climate model is impossible — two runs of
the *same* Fortran that differ only by round-off produce different multi-year
means. The correct standard is **statistical reproduction**: the jsca-vs-Isca
difference in a climatology must sit within Isca's own internal-variability
envelope. `jsca` enforces this at two levels:

- **Module level (Tier-1):** every ported routine is checked against golden
  input/output fixtures dumped from the instrumented Fortran, at `rtol` between
  `1e-14` and `1e-11`.
- **Climate level (Tier-3):** an ensemble-mean equivalence test (Student-t
  against the Fortran's month-to-month spread, FDR-controlled, with
  practical-significance floors) decides whether two climatologies are
  statistically indistinguishable.

Once fidelity is proven, the JAX rewrite pays off in ways the Fortran cannot:
`float64` throughout, a `jit`/`vmap`/`scan`-composable step function, trivial
ensemble parallelism, GPU portability, and end-to-end differentiability — the
model becomes a function you can autodiff.

## Design rules

Every module in `jsca` obeys the same discipline (see `CLAUDE.md` for the
canonical statement):

```{list-table}
:header-rows: 1
:widths: 30 70

* - Rule
  - What it means
* - **Faithful means faithful**
  - Port Isca's algorithm, constants, and call order. Cite the Fortran source
    (file and lines) in the docstring. Never silently "fix" or modernize Fortran
    behaviour — reproduce it and flag the oddity. Any deliberate deviation (e.g.
    LAPACK for the matrix inverse) is documented with its fixture-proven
    tolerance.
* - **Fixtures from Fortran only**
  - Golden data comes from the real Fortran — never generated, patched, or
    "refreshed" from Python output.
* - **JAX discipline**
  - `float64` (enabled by `import jsca`); pure functions of
    `(config, params, state)`; no module-level state; everything on the step
    path `jit`/`vmap`/`scan`-safe; configs static/hashable; numeric tables
    precomputed into `params`.
* - **Conventions**
  - Latitude south→north; spectral storage `(m, n)` with total wavenumber
    `l = m + n`; transforms act on `(..., nlat, nlon)`; column physics put the
    level axis **last** (`k = 0` top … `K` surface).
```

## Anatomy of a jsca model

Every configuration is built from three kinds of object:

- **config** — a frozen, hashable dataclass of static choices (resolution,
  scheme switches, physical constants). Safe to close over in `jit`.
- **params** — a pytree of precomputed `float64` numeric tables (Gaussian
  weights, Legendre functions, damping coefficients, vertical-coordinate
  coefficients, the LCL lookup table, …). Built once at init.
- **state** — the prognostic pytree that the step function advances (spectral
  vorticity/divergence/temperature/log-surface-pressure, the grid humidity
  tracer, surface temperature, bucket depth, and the leapfrog time levels).

A model step is a **pure function** `step(config, params, state) -> state`. The
whole time loop composes with `lax.scan`, so a multi-year integration is one
compiled kernel with no host synchronization inside it.

## State of the port

The dynamical core and the four flagship configurations are complete and
validated.

```{list-table}
:header-rows: 1
:widths: 34 22 44

* - Component
  - Status
  - Notes
* - {py:mod}`jsca.grid`
  - ✅ fixture-validated
  - Gaussian grid, GFDL-normalized associated Legendre, spectral transforms,
    Laplacian / hyperdiffusion.
* - {py:mod}`jsca.dycore`
  - ✅ fixture-validated
  - The **full GFDL spectral core**: leapfrog/RAW (incl. two-level split),
    matrix inversion, pressure & geopotential, spectral damping, semi-implicit
    solve, spherical operators, finite-volume advection, water borrowing, global
    integrals, and the `spectral_dynamics` assembly.
* - Grey radiation
  - ✅ fixture-validated
  - `two_stream_gray_rad` — `rad_scheme='frierson'` and `'byrne'`; `do_seasonal`
    seasonal + diurnal insolation via the ported `astronomy` module.
* - Convection
  - ✅ fixture-validated
  - `SIMPLE_BETTS_MILLER`, `FULL_BETTS_MILLER`, `DRY` (Schneider–Walker), `NONE`.
* - Condensation & surface
  - ✅ fixture-validated
  - `lscale_cond`, `monin_obukhov`/`surface_flux`, `diffusivity` (both `do_simple`
    branches), `vert_diff`, `mixed_layer`, `damping_driver`, bucket hydrology.
* - {py:mod}`jsca.model.held_suarez`
  - ✅ climatology-validated
  - Dry HS94; 0.0 % of points differ from Isca beyond internal variability.
* - {py:mod}`jsca.model.frierson`
  - ✅ climatology-validated
  - Moist aquaplanet; the **full 3D run matches Isca** (jet 38.0 vs 38.1 m/s).
* - {py:mod}`jsca.model.column`
  - ✅ validated vs Isca
  - Single-column model; day-40 SST RMSD 0.012 K vs Isca.
* - {py:mod}`jsca.model.bucket_model`
  - ✅ climatology-validated
  - Land + Manabe bucket; T21 pattern correlations 0.96–1.00 vs Isca.
```

The current development focus is **swappable physics options** — new radiation,
convection, and surface schemes — each validated in the single-column model
(fast) before it runs in the full 3D model, and each opt-in so the validated
runs stay byte-for-byte unchanged.

## The pinned reference

All ports and fixtures refer to one Isca commit —
`ExeClim/Isca@a290bc376d84d0ee83adbb80eb374b9f629c3534` (master, 2026-01-30). Do
not port from a different Isca version; the fixture tolerances assume this exact
source.
