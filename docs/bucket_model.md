# Land + bucket-hydrology model (`jsca.model.bucket_model`)

This is the `land_option='input'` companion to the Frierson aquaplanet
(`jsca.model.frierson`): the same validated dynamical core, grey radiation and
column physics, run over a **land mask** with a Manabe-style **soil-moisture
bucket**. It reproduces Isca's `exp/test_cases/bucket_hydrology` configuration —
**with grey radiation** (the shipped test case uses RRTM; grey radiation is the
jsca equivalent) and **realistic continents** — for a jsca-vs-Isca comparison at
T21.

## What "land" changes

Land enters through three switches, each ported and Tier-1 fixture-validated in
its own module before being wired together here:

| Piece | Module | Isca source | Fixture |
|---|---|---|---|
| Dry/wet **evaporation switch** + β-ramp | `physics/surface_flux.py` | `surface_flux.F90` L448-643 | `test_surface_flux_bucket_fixtures.py` |
| Land **heat capacity** & **albedo** prefactors | `physics/mixed_layer.py` | `mixed_layer.F90` L437/L554 | `test_mixed_layer_land.py` |
| Soil-moisture **reservoir stepping** (leapfrog+RAW+runoff) | `physics/bucket.py` | `idealized_moist_phys.F90` L1401-1428 | `test_bucket_fixtures.py` |
| Idealized-**continents mask** | `model/land.py` | `land_generator_fn.py` L84-105 | `test_land_mask.py` |

The reservoir is a prognostic `bucket_depth` [m of water], added to the model
state as a `(nlat, nlon, 2)` field (leapfrog time level last, same convention as
the humidity tracer). Each timestep, after the atmosphere leapfrog:

```
dt_bucket   = depth_change_cond + depth_change_conv - depth_change_lh
bucket_depth = leapfrog+RAW(bucket_depth, dt_bucket)   # robert=0.04, raw=0.53
bucket_depth = max(bucket_depth, 0)                    # never negative
bucket_depth = min(bucket_depth, max_bucket_depth_land)  # runoff, land only
```

where the two precipitation depths come from the column physics
(`depth_change_{cond,conv} = rain_{ls,conv} / dens_h2o`, Isca L1021/L911) and the
evaporation depth from the surface-flux bucket path
(`depth_change_lh = flux_q · Δt / dens_h2o`). Over an **empty** land bucket the
surface flux sets the surface humidity to the air value, so there is no
evaporation; below 0.75·capacity the evaporation is linearly β-ramped.

## The land mask: continents, not the shipped square

The user asked to reproduce the bucket test case with **realistic continents**.
Worth flagging: Isca's shipped `bucket_hydrology/input/land.nc` is actually a
**square block** of land (`land_mode='square'`, ~±33° lat × 0–95° lon, ~9.7 %
land), *not* continents. `jsca.model.land.continents_land_mask` instead ports the
`land_mode='continents'` geometry from `land_generator_fn.py` — the
Sauliere-2012 continent set (N/S America, Eurasia, Africa, Australia, India, SE
Asia, ~18.8 % land). It is validated bit-for-bit against that Fortran-adjacent
Python run on Isca's own T42 grid, then evaluated on the T21 Gaussian grid.

For the Stage-4 T21 validation, **both** jsca and Isca are driven from this
continents mask (written to a `land.nc` Isca reads), so the comparison stays
like-for-like rather than comparing continents against the shipped square.

## Configuration (`build_bucket_model` defaults)

Mirrors the `bucket_hydrology` namelist, with grey radiation substituted for
RRTM:

- **Vertical**: 40 levels, `uneven_sigma`, `surf_res=0.2`, `scale_heights=11`,
  `exponent=7`.
- **Convection**: `SIMPLE_BETTS_MILLER`; large-scale condensation on.
- **Mixed layer**: `depth=20 m`, `albedo_value=0.25`,
  `land_h_capacity_prefactor=0.1`, `land_albedo_prefactor=1.3`.
- **Bucket**: `max_bucket_depth_land=2 m`, `init_bucket_depth_land=1 m`, ocean
  reservoir effectively infinite (always wet).
- **Surface**: roughness `2e-4 m`, `do_simple`, `use_virtual_temp=False`,
  `constant_gust=0`.
- **Dynamics**: `robert_coeff=0.03`, `damping_order=4`.

## Validation status

- **Kernels**: each land piece + the continents mask is golden-fixture-validated
  (table above).
- **Assembly**: `tests/test_bucket_model.py` is a stability + water-budget smoke
  test — a short integration stays finite and physical (land reservoir in
  `[0, capacity]`, ocean always wet, humidity ≥ 0), and the precipitation →
  reservoir-depth conversion matches Isca's `rain/dens_h2o` exactly.
- **Aquaplanet unchanged**: the land/bucket wiring in `idealized_moist_phys` is
  opt-in (defaults leave `land=None`), so the validated Frierson run is
  byte-for-byte identical.
- **Next (Stage 4)**: the T21 continents climatology compared against a pinned
  Isca bucket run (bucket_depth, precip, t_surf).
