# Regenerating the spectral_dynamics step-by-step fixture

`tests/fixtures/spectral_dynamics_step_reference.npz` is an **end-to-end**
Tier-1 fixture: the prognostic spectral state after each step of Isca's
`spectral_dynamics` time loop, from an instrumented full-model run. It validates
jsca's assembled Held-Suarez step (`jsca.model.held_suarez.step`) against the
Fortran loop, one step at a time (`tests/test_spectral_dynamics_step_fixtures.py`).

`spectral_dynamics.F90` is un-stubbable (it needs the whole model), so unlike the
per-routine fixtures there is no standalone driver — the harvest is a patched,
serial full-model run.

## Recipe

1. Build the pinned Isca (`ExeClim/Isca` commit `a290bc3`) for the dry core, as in
   `baseline/PINNED.md` (gfortran flags, `git init`, `GFDL_ENV=docker`).
2. Drop `jsca_dump.F90` into `src/atmos_spectral/model/` and add
   `atmos_spectral/model/jsca_dump.F90` to `src/extra/model/dry/path_names`.
3. Apply `spectral_dynamics_step_instrument.patch` (this directory) to
   `src/atmos_spectral/model/spectral_dynamics.F90`. It adds, guarded by a global
   step counter (`jsca_ndump` steps):
   - **once**: config scalars, `pk`/`bk`/`coriolis`/`surf_geopotential`, and the
     initial spectral state (`vors,divs,ts,ln_ps` at previous & current, real+imag);
   - **each step, before `initialize_corrections`**: `delta_t`, `prev_eq_cur`, and
     the incoming physics tendencies `dt_ug,dt_vg,dt_tg,dt_psg`;
   - **each step, after the time-pointer swap**: the prognostic spectral state at
     the new current (= future) and the previous slot (real+imag);
   - (correction scalars in `compute_corrections` — used for debugging the
     energy-correction residual; harmless to keep.)
4. Recompile. Run the dry **Held-Suarez** test case at **T21, 15 levels**,
   `uneven_sigma` (`scale_heights=6`, `surf_res=0.5`, `exponent=7.5`),
   `damping_order=4`, `dt_atmos=600`, **`num_cores=1`** (serial ⇒ the dumped
   spectral fields are global/undecomposed), `days=1`, with
   `JSCA_DUMP_DIR` set and `mpirun_opts='... -x JSCA_DUMP_DIR'`.
5. Convert the first 8 steps to the compact `.npz`: combine each `*_re`/`*_im`
   pair into a complex array; keep the initial `(prev,cur)` state, the per-step
   `delta_t`/`prev_eq_cur`, and the per-step output `vors/divs/ts/lnps`.

Fortran spectral storage `(m, n)` = jsca `(m, n)` (verified: shapes and
`coriolis`/`pk`/`bk` match exactly). Grid forcing is `(lon, lat, k)` and needs a
`(lon,lat)->(lat,lon)` transpose to jsca's `(lat, lon, k)` (latitude is S→N in
both). The fixture is kept at low resolution purely so it is small enough to
commit.
