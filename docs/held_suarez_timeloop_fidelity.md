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

## Still open

A subtler, **`dt=600`-only** growing mode remains: even driven by Isca's exact
forcing, jsca's trajectory diverges from Isca's, growing ~1.5×/step from
machine-precision (temperature leads, then vorticity/divergence). It is **not**
the energy correction (injecting Isca's exact global-mean correction does not
change it) and jsca is stable at `dt ≤ 300`. The remaining suspects are the
semi-implicit vertical solve at large `dt` (jsca uses LAPACK where Isca uses
Gauss-Jordan — a documented ~1e-11 deviation that could excite a marginal mode
near the stability edge) or another small operator difference. The step fixture
is the tool to bisect it; the coarse-resolution fixture committed here stays well
clear of the mode, so it passes cleanly and guards the fixes above.

Practically: the Held-Suarez climatology is `dt`-insensitive once stable, so the
jsca-vs-Isca comparison can run at `dt=300` while the `dt=600` mode is chased.
