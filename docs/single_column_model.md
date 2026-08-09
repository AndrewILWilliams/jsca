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

Against the canonical `column_test.py` reference:

| diagnostic | agreement |
|---|---|
| day-40 SST | jsca vs Isca (Δ 0.03 K) |
| day-40 precip | Δ ~0.13 mm/day (~4%) |
| day-40 T profile | RMSD 0.21 K |
| day-40 q profile | RMSD 0.20 g/kg |
| SST trajectory (40 d) | RMSD 0.02 K |

The SST trajectory tracks Isca to hundredths of a kelvin. Getting there meant
reproducing **Isca's deliberately unstable slab initialisation** — `t_surf =
init_temp + 1 K` (`idealized_moist_phys.F90` L643, "to allow moisture to quickly
enter the atmosphere avoiding problems with the convection scheme"). This +1 K, not
the mechanisms first guessed (`constant_gust`, `do_lcl_diffusivity_depth`, both since
shown to be negligible here), was the dominant difference: without it the SST
trajectory is offset ~1 K early, decaying to ~0.3 K by day 40; with it the SST RMSD
drops from **0.55 K to 0.02 K**. (Diagnosis: Isca's per-step `delta_t_surf` matched
jsca's to 1e-4 from step 1 — only the *initial* SST differed.) A second, minor config
match: `lscale_cond do_evap=False` (column_test disables rain re-evaporation), now
threaded through `idealized_moist_phys`.

jsca now runs the **full canonical column config**: `surface_flux
use_virtual_temp=True` (the `d608*q` virtual-temperature correction to the surface-layer
stability) and `lscale_cond do_evap=False` are both ported/threaded. The
`use_virtual_temp` path is golden-fixture-validated against Isca to 1e-6
(`tests/test_surface_flux_fixtures.py`, from `dump_surface_flux_vt_reference.F90`), and
its effect on the fluxes matches Isca's to 1e-4.

A residual day-40 **profile** difference remains: RMSD ~0.21 K / 0.20 g/kg at the
global-average column, ~0.37 K / ~0.45 g/kg in the moist tropics, ~0.03 K at high
latitudes. Interestingly this is **not** a config toggle — with every namelist option
now matched, turning `use_virtual_temp` on changed the column T profile by only
~0.04 K RMSD (vs ~0.12 K in Isca's own on/off test), because the perturbation is
amplified differently by the base-state difference itself. So the residual is a small
**per-step numerical/structural difference**, largest where humidity is largest.

**Localising it (golden step fixture, started).** Instrumenting the running Isca
column and dumping the `qe_moist_convection` I/O at a real step (step 600) settles
one suspect: **convection is exact.** Fed Isca's true instantaneous column state,
jsca's convection reproduces Isca's rain to **+0.00%**, the same `klzb`/`convflag`,
and the T/q tendencies to the `sat_vapor_pres` es floor (~2e-7) —
`tests/test_column_convection_step_fixtures.py`, recipe
`fortran_instrumentation/column_convection_step_recipe.md`. So the few-percent
daily-mean precip difference is **not** a convection-scheme error (a mean-state
convection call over-rains by ~6% only because convection is nonlinear and the
daily-mean profile is smoother than the instantaneous ones it acts on); it is a
downstream consequence of the residual profile difference. The vertical structure of
that residual — near-perfect above the convective top (~400 hPa), growing in the
**boundary layer** (levels 25-30) — points the remaining search at the boundary-layer
diffusion / `vert_diff` chain. Dumping those stages the same way is the next step
(issue #43).

### CI regression gate

`tests/test_column_vs_isca.py` (single column) and `tests/test_column_sweep_vs_isca.py`
(five latitudes) run this comparison on every commit and assert jsca stays within
tolerance of the Isca trajectory — after the `t_surf` fix, SST RMSD < 0.1 K
(measured 0.02), and the profile tolerances (T-profile RMSD < 0.3 K, q < 0.3 g/kg;
< 0.5 K / 0.6 g/kg across the sweep to cover the moist tropics) sized to the
`use_virtual_temp` gap — so a physics regression fails CI rather than being
discovered later. **CI never needs Isca itself**: it validates against committed
golden trajectories
(`baseline/reference/column_scm_isca_t264.npz`, `…_sweep.npz`; numpy-only, distilled
from the raw Isca NetCDF), the same posture as the frierson climatology references.
The remaining tropical-profile tolerance headroom is the per-step residual above;
the golden step fixture (#43) is the checkpoint for tightening it further.

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
