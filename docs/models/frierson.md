# Frierson moist aquaplanet

The Frierson (2006) idealized moist aquaplanet is `jsca`'s flagship moist
configuration: the validated spectral core coupled to grey radiation, simplified
Betts–Miller convection, large-scale condensation, a bulk boundary layer, surface
fluxes, and a slab ocean — everything on a water-covered planet with no land.

:::{admonition} Source
:class: note
{py:mod}`jsca.model.frierson`, with column physics from
{py:mod}`jsca.model.idealized_moist_phys`.
:::

## What it adds over Held–Suarez

Relative to the dry benchmark, the moist model:

- makes **specific humidity $q$ a prognostic grid tracer**, advected by the
  finite-volume operators and stepped with a RAW leapfrog (with the global
  water-conservation correction);
- replaces the Newtonian/Rayleigh forcing with the full
  **`idealized_moist_phys`** column-physics driver (the ten-step call order in
  {doc}`../physics/index`);
- adds a prognostic slab-ocean surface temperature `t_surf`.

## Configuration

`build_frierson` defaults mirror Isca's `frierson_test_case`:

```{list-table}
:header-rows: 1
:widths: 30 20 50

* - Setting
  - Default
  - Note
* - spectral truncation
  - T42
  - `num_fourier=42`, 64×128 grid.
* - vertical
  - 25 levels
  - The Frierson pure-sigma `input` coordinate table.
* - timestep `dt`
  - 720 s
  - Leapfrog interval is `2·dt`.
* - `robert_coeff`
  - 0.03
  - RAW filter strength.
* - `damping_order`
  - **4** ($\nabla^8$)
  - Frierson uses order 4, *not* the Isca default of 2 — order 2 over-damps the
    eddies (see {doc}`../governing_equations/diffusion`).
* - mixed-layer depth
  - 2.5 m
  - Slab ocean.
* - albedo
  - 0.31
  -
```

```python
from jsca.model import frierson as F

model  = F.build_frierson(num_fourier=42)
state0 = F.initial_state(model)   # quiescent 264 K, ps = 1e5, q = 2e-6, meridional SST

state, clim = F.integrate_climatology(
    model, state0, spinup_steps=..., avg_steps=..., cold_start=True,
)
# clim carries time-mean u, v, T, q, t_surf, precip
```

The initial condition is ported faithfully from Isca
(`spectral_initialize_fields.F90`, `mixed_layer.F90`, `frierson_test_case.py`):
quiescent, isothermal 264 K, `initial_sphum = 2e-6`, and a meridional SST
$285 - 40\,(3\sin^2\phi - 1)/3$. The one deliberate deviation is a $\sim10^{-4}$ K
symmetry-breaking temperature perturbation that stands in for the MPI-domain
round-off by which Isca breaks exact zonal symmetry (which `float64` jsca cannot
reproduce).

## Validation

The **full 3D run is validated against Isca** at T42 and T21. On the strictest
like-for-like test (jsca on Isca's exact 64×128 grid and initial condition, days
200–300), jsca reproduces the double eddy-driven jet, the ITCZ precipitation
peak, and the midlatitude storm-track rain:

```{list-table}
:header-rows: 1
:widths: 34 22 22 22

* - Field
  - correlation
  - RMSE
  - bias (jsca − Isca)
* - specific humidity $q(\phi, p)$
  - **0.9996**
  - 0.14 g/kg
  - +0.02 g/kg
* - precipitation
  - **0.9954**
  - 0.40 mm/day
  - −0.06 mm/day
* - temperature $T(\phi, p)$
  - **0.9975**
  - 2.1 K
  - −0.86 K
* - surface temperature `t_surf`
  - **0.9991**
  - 1.8 K
  - −0.69 K
* - zonal wind $u(\phi, p)$
  - **0.942**
  - 4.0 m/s
  - +1.9 m/s
```

The eddy-driven jet peaks at **38.0 m/s vs Isca 38.1 m/s**. A Tier-3 ensemble
equivalence test at T21 returns `fail_fraction = 0.0 %` on every field. The full
story — including the water-conservation and damping-order bug fixes that got
here, and a +2 % solar-constant climate-*change* comparison — is in
{doc}`../validation/index`.

## Performance

At 64×128 the model runs about **178 ms/step**, roughly 1.6× faster than
single-core Isca, with no host synchronization inside the scanned time loop.
