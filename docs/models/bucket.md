# Land + bucket-hydrology model

The bucket model adds **land and Manabe soil-moisture hydrology** to the Frierson
moist stack: the same validated dynamical core, grey radiation, and column
physics, run over a land mask with a prognostic soil-moisture reservoir. It
reproduces Isca's `bucket_hydrology` test case — with grey radiation substituted
for RRTM and realistic continents.

:::{admonition} Source
:class: note
{py:mod}`jsca.model.bucket_model`, adding {py:mod}`jsca.physics.bucket` and
{py:mod}`jsca.model.land` to the Frierson stack. **Reference:** Manabe (1969).
:::

## What "land" changes

Land enters through three switches, each ported and fixture-validated in its own
module before being wired together (see {doc}`../physics/surface`):

```{list-table}
:header-rows: 1
:widths: 40 30 30

* - Piece
  - Module
  - Isca source
* - Dry/wet evaporation switch + β-ramp
  - `physics/surface_flux.py`
  - `surface_flux.F90`
* - Land heat capacity & albedo
  - `physics/mixed_layer.py`
  - `mixed_layer.F90`
* - Soil-moisture reservoir stepping
  - `physics/bucket.py`
  - `idealized_moist_phys.F90`
* - Idealized-continents mask
  - `model/land.py`
  - `land_generator_fn.py`
```

The reservoir is a prognostic `bucket_depth` (metres of water) added to the model
state as a `(nlat, nlon, 2)` field (leapfrog time level last, like the humidity
tracer). Each step, after the atmosphere leapfrog, it is filled by precipitation
and drained by evaporation, capped at capacity as runoff (the equations are in
{doc}`../physics/surface`).

## Configuration

`build_bucket_model` mirrors the `bucket_hydrology` namelist with grey radiation:
40 levels (`uneven_sigma`), `SIMPLE_BETTS_MILLER` convection, mixed-layer depth
20 m with `land_h_capacity_prefactor=0.1` and `land_albedo_prefactor=1.3`,
`max_bucket_depth_land=2 m`, `init_bucket_depth_land=1 m`, `damping_order=4`.

The land/bucket wiring is **opt-in** — the defaults leave `land=None`, so the
validated Frierson aquaplanet run is byte-for-byte unchanged.

```python
from jsca.model import bucket_model as B

model  = B.build_bucket_model(land=land_mask)   # continents mask on the model grid
state0 = B.initial_state(model)
state, clim = B.integrate_climatology(model, state0, spinup_steps=..., avg_steps=...)
```

## Validation

The full 3D bucket model is validated against a pinned-Isca run of the *same*
configuration at T21 (32×64, 40 levels) over the *same* continents mask. Both
models ran 12 × 30-day months from a matched cold start; the comparison is the
time-mean over the last 6 months:

```{list-table}
:header-rows: 1
:widths: 34 22 22 22

* - Field
  - bias
  - RMSE
  - correlation
* - `bucket_depth` (land)
  - +0.013 m
  - 0.143 m
  - **0.96**
* - precip
  - −0.11 mm/day
  - 0.86 mm/day
  - **0.97**
* - `t_surf`
  - −0.002 K
  - 1.10 K
  - **1.00**
* - `t_surf` (land)
  - −0.045 K
  - 1.39 K
  - **0.99**
```

Both models produce the classic Manabe soil-moisture pattern — driest land over
the subtropics, wettest over the midlatitude storm tracks — the ITCZ
precipitation band, and warm continental surface temperatures. The residual
differences are small-scale and unbiased, the same standard of agreement as the
Frierson aquaplanet. See {doc}`../validation/index` for the full figure.
