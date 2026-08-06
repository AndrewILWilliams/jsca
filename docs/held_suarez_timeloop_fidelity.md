# Held-Suarez time-loop fidelity: what the step-by-step fixture found

The dry Held-Suarez run blew up at the full Isca benchmark resolution (T42 L25,
`dt=600 s`) while it was stable at coarser settings. Rather than reason about it
from physics, we built a **step-by-step Fortran fixture**: an instrumented Isca
`spectral_dynamics` dumps its prognostic spectral state after every step, and
jsca is run from the identical initial state and compared step for step
(`tests/test_spectral_dynamics_step_fixtures.py`; recipe in
`fortran_instrumentation/spectral_dynamics_step_recipe.md`).

## What matched

Driven by Isca's exact forcing, jsca's dynamical core reproduces Isca's
vorticity and divergence to **~1e-13 per step** — the semi-implicit gravity-wave
correction, advection, leapfrog+RAW filter, and hyperdiffusion are faithful. The
semi-implicit *is* controlling gravity waves (an earlier "the semi-implicit is
ineffective" hypothesis was wrong).

## Bugs the fixture caught (now fixed)

1. **Physics forcing was applied to the wrong time level.** Isca's driver passes
   `ug/vg/tg(previous)` to `hs_forcing` (`atmosphere.F90` L304-311); jsca passed
   the *current* level. For Rayleigh friction — a damping term — evaluating it on
   the current level in a leapfrog feeds the computational mode instead of
   damping it. Against the Fortran this was an ~8% error in the momentum forcing
   (machine-zero once corrected). This is the leading cause of the
   high-resolution instability. Pressure levels stay at `current`, as in Isca.

2. **Cold-start step used the wrong interval.** Isca's first step from rest has
   `previous == current` and uses `delta_t = dt` (a forward step), then `2 dt`
   thereafter. jsca used `2 dt` throughout. `integrate(..., cold_start=True)` now
   does the forward step first.

3. **Energy-correction reference** now advances the previous state by the physics
   forcing over `delta_t`, matching `initialize_corrections` (F90 L1373-1379).

A fourth latent gap — `build_dynamics_params` not threading the `uneven_sigma`
stretching parameters into the vertical coordinate — was fixed alongside.

## The `dt=600` growing mode — found and fixed

A subtler, `dt=600`-only mode initially remained: even with Isca's exact forcing,
jsca's trajectory grew ~1.5×/step from machine precision. Bisecting the error
against Isca in `(wavenumber, level, field)` space localized it immediately: the
growth lived **entirely in the total-wavenumbers `l = m + n > M`**, which are
outside the T_M triangular truncation and are **exactly zero in Isca**. jsca was
leaving them non-zero.

The cause: Isca truncates the prognostic tendencies to the triangle (`l <= M`)
inside `trans_grid_to_spherical` (default `do_truncation=.true.`), whereas jsca's
`grid_to_spectral` keeps the `l = M+1` storage diagonal — correct for the
`d/dmu` derivative paths (`four_in_one`, the spherical operators), but it leaks
`l = M+1` into `dt_ts`, `dt_ln_ps` and (via the Laplacian of `phi + KE`)
`dt_divs`. Those modes are unphysical grid-scale content that nothing damps; at
high resolution / large `dt` they grow exponentially and blow the model up. At
`dt <= 300` (or T21) they grow slowly enough to stay bounded — which is why the
model *looked* fine there.

**Fix:** triangular-truncate the prognostic tendencies to `l <= M` at the end of
`compute_tendencies` (and the initial fields in `initial_state`), matching Isca.
Per-step agreement with Isca returns to machine precision (vor/div ~1e-15), and
the model is **stable at the full Isca benchmark config (T42 L25, `dt=600 s`)** —
it now spins up smoothly where it previously NaNed by day 2.

The only residual is the ~1e-5 global-mean `(0,0)` energy-correction term (does
not grow, does not affect stability), left as a separate minor item.
