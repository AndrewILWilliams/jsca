# The single-column model (SCM) port

This documents the jsca port of Isca's single-column model (`src/atmos_column`),
implemented in `jsca/model/column.py`. It is the Tier-2 physics-chain test harness
called for in the scoping doc (§4.4): the full column physics stepped over many
timesteps **with the dynamical core bypassed**, so a jsca-vs-Isca comparison
isolates the physics from spectral-dynamics chaos. Cite McKim et al. (2024) for the
Isca SCM.

## What the SCM is

Isca's `column_mod` is a **drop-in replacement for `spectral_dynamics`** in the solo
driver (`atmosphere.F90`, selected by the `-DCOLUMN_MODEL` compile flag). The
per-step physics — convection → large-scale condensation → grey radiation → surface
fluxes → boundary-layer diffusion → slab ocean — is identical to the full model
(`idealized_moist_phys`, already ported and assembled in
`jsca/model/idealized_moist_phys.py`). What the column replaces is everything
*dynamical*:

| | Full model (`frierson.py`) | Single column (`column.py`) |
|---|---|---|
| horizontal transport | spectral advection | none |
| semi-implicit gravity-wave solve | yes | none |
| spectral damping / sponge | yes | none |
| mass / energy / water global corrections | yes | none |
| winds `u, v` | prognostic | **prescribed, fixed** |
| surface pressure `ps` | prognostic (continuity) | **fixed** |
| temperature `T` | spectral + physics | **grid leapfrog of physics `dt_tg`** |
| humidity `q` | tracer advection + physics | **grid leapfrog of physics `dt_qg`** |
| slab SST `t_surf` | mixed layer (in physics) | mixed layer (in physics) |

So one SCM step is: run the physics on the previous time level, then leapfrog **only
T and q** forward with the physics tendencies. The momentum tendency from the
boundary-layer diffusion is *computed and discarded* — the SCM's stated assumption
is that "the dynamics" would restore the prescribed surface wind, so the net
d(u)/dt = 0 (`column.F90` returns `ug/vg` from the unchanged previous slot,
L358-360). Surface pressure never changes, so `p_half`/`p_full` are constant in time.

## Faithfulness (CLAUDE.md rules 1-2)

* **Leapfrog.** The SCM's `leapfrog_3d_real` (`column.F90` L756-799) is arithmetically
  identical, line by line, to the already-ported `jsca.dycore.leapfrog.leapfrog`
  (the combined Robert–Asselin–Williams form). Note the SCM's *real* variant **does**
  apply `raw_filter_coeff` on the current-level update, unlike the dycore's
  `leapfrog_2level_A_3d_real` quirk — so `leapfrog` (not the `_real` split) is the
  faithful match. The SCM defaults `robert_coeff = 0`, `raw_filter_coeff = 1`
  (`column_nml`): with no dynamics there is no gravity-wave noise for the filter to
  control.

* **`q_decrease_only`.** The optional stratospheric humidity clamp (`column.F90`
  L782-792) sweeps levels upward, capping each at the level below it. With the level
  axis last (k = 0 top … K-1 surface) applied per column, that is exactly a reverse
  cumulative minimum, implemented in `jsca.dycore.leapfrog.apply_q_decrease_only`.
  *Documented deviation:* the Fortran takes its clamp **decision** from the first
  column `q(1,1,k)` but assigns to all columns — only self-consistent for a genuine
  single column. jsca applies the clamp independently per column (identical for the
  single-column case the SCM is built for; more sensible for multi-column runs).

* **Initial condition.** `initial_state` ports `column_initialize_fields.F90`
  L74-79: `u = v = surface_wind/√2` at the bottom level (zero aloft), uniform T,
  `ps = exp(ln p_ref − Φ_s/(R_d T₀))`, uniform `sphum`. The slab starts uniform at
  `tconst` (`prescribe_initial_dist = False`).

* **Geopotential time level.** The physics sees geopotential heights computed from
  the **current**-level temperature, matching `atmosphere.F90`'s column flow (it
  recomputes `z_full/z_half(current)` from `tg` at the end of each step and hands
  those to the next physics call). Pressures use the fixed `ps`.

Every physics kernel is golden-fixture-validated against Isca to machine precision;
the SCM is their *assembly*, gated (like the `frierson.py` assembly) by a
stability/invariant smoke test (`tests/test_column.py`). A machine-precision golden
column-step fixture needs a full Isca build; the driver stub for the new
initial-condition arithmetic is
`fortran_instrumentation/dump_column_init_reference.F90` (fixtures pending).

## Resolved: the vertical coordinate (was a flagged ambiguity)

The canonical Isca column namelist
(`exp/test_cases/column_test_case/column_test.py`) sets **both**
`column_nml:num_levels = 31` (with the default `vert_coord_option = 'even_sigma'`)
**and** an explicit 25-level `vert_coordinate_nml`. Because `even_sigma` never reads
that namelist, the 26-entry `pk`/`bk` would be silently ignored.

**Confirmed by a real Isca column run** (2026-08; output archived at
`baseline/reference/column_scm_isca_daily_t264.nc`): Isca runs **31 even-sigma
levels** (`bk = [0, 1/31, …, 1]`) and ignores the 25-level table, exactly as
hypothesised. `build_column()` therefore still defaults to the 25-level Frierson
coordinate (the levels the physics kernels were fixture-validated on), and
`scripts/compare_column_scm.py` reads Isca's actual `pk`/`bk` from the reference
file so the comparison is on Isca's real 31 even-sigma levels. To reproduce Isca's
default column grid directly, pass
`vert_coord_option='even_sigma', num_levels=31, pk=None, bk=None`.

A second thing the run pinned down: with `prescribe_initial_dist = False` and no
restart, Isca's `mixed_layer` seeds the slab SST from the **lowest model-level
temperature** (= `initial_temperature`, 264 K), not `tconst` (285 K).
`initial_state` now defaults `t_surf` to `initial_temperature` to match.

## Validation against Isca (Tier-2)

A 40-day single-column spin-up, jsca vs a real Isca run on Isca's own 31 even-sigma
levels, same latitude / timestep / cold-start IC and the canonical `column_test.py`
physics (`scripts/compare_column_scm.py`, figure
`docs/figures/column_scm_vs_isca.png`):

| diagnostic | agreement |
|---|---|
| day-40 SST | jsca 287.80 K vs Isca 288.02 K (Δ 0.22 K) |
| day-40 precip | jsca 3.05 vs Isca 3.14 mm/day (~3%) |
| day-40 T profile | RMSD 0.38 K |
| day-40 q profile | RMSD 0.12 g/kg |
| SST trajectory (40 d) | RMSD 0.55 K |

The equilibrium T profile overlies Isca almost exactly; the SST/precip trajectories
track closely with jsca running slightly cool for the first ~15 days before
converging. The residual is consistent with two documented differences, not a
porting bug: (i) jsca's `constant_gust = 0` vs Isca's stateful `vert_turb` gustiness
during the cold-start transient, and (ii) `do_lcl_diffusivity_depth = True` in
`column_test` (Isca sets the PBL depth from the convective LCL) which jsca's
Richardson-based `diffusivity` does not yet implement. Closing those is tracked in
issue #43, along with the near-bitwise golden step fixture.

### CI regression gate

`tests/test_column_vs_isca.py` runs this comparison on every commit and asserts jsca
stays within tolerance of the Isca trajectory (T-profile RMSD < 0.6 K, SST RMSD
< 0.8 K, q-profile RMSD < 0.3 g/kg, final precip within 0.3 mm/day) — so a physics
regression fails CI rather than being discovered later. **CI never needs Isca
itself**: it validates against the committed golden trajectory
`baseline/reference/column_scm_isca_t264.npz` (numpy-only, distilled from the raw
Isca NetCDF), the same posture as the frierson climatology references. Tightening
these tolerances is the checkpoint for closing the two config gaps above.

### Reproducing / regenerating the Isca reference

Everything needed to rebuild the Isca side is committed, so the reference can be
regenerated whenever the physics config or the pinned Isca changes:

```bash
bash scripts/build_isca_column.sh                 # toolchain + pinned Isca + patch
python scripts/run_isca_column_reference.py 40    # run Isca -> atmos_daily.nc
cp <run>/atmos_daily.nc baseline/reference/column_scm_isca_daily_t264.nc
python scripts/distill_column_reference.py        # NetCDF -> committed .npz
python scripts/compare_column_scm.py <atmos_daily.nc>   # refresh the figure
```

This keeps the "reproduce Isca if we want to" path live while CI itself stays
Isca-free.

## Usage

```python
from jsca.model import column as C

m  = C.build_column()                 # 1 column, Frierson 25-level, global-avg lat
s0 = C.initial_state(m)               # cold start (T=264 K, q=1e-3, u_surf=5 m/s)
s, clim = C.integrate_climatology(m, s0, spinup_steps=..., avg_steps=..., cold_start=True)
# clim: time-mean 'temp','sphum' (nlat,nlon,K), 't_surf','precip' (nlat,nlon)
```

`build_column` also runs a small set of **independent** columns via `latitudes` /
`longitudes` (each responds to its own insolation; there is no horizontal coupling).

## Not ported

* `num_steps > 1` sub-stepping inside a single `column` call (Isca default is 1).
* Restart I/O, diagnostics manager, `graceful_shutdown`, JSON logging, the
  `hs_forcing` (dry) column path (the SCM is wired to the moist physics; the dry
  path is a small future addition), and the `global_average` insolation trick (the
  single-column test uses `lat_value = arcsin(1/√3)` instead — carried as
  `GLOBAL_AVERAGE_LAT_DEG`).
