# Single-column model (SCM)

The single-column model runs the **full column physics with the dynamical core
bypassed** — Isca's `atmos_column` driver. It isolates the physics from
spectral-dynamics chaos, which makes it the fast, exact validation bench for new
physics options: a jsca-vs-Isca column comparison localizes any discrepancy to a
single scheme.

:::{admonition} Source
:class: note
{py:mod}`jsca.model.column` (ports `src/atmos_column`). **Reference:** McKim et
al. (2024) for the Isca SCM.
:::

## What the column replaces

The per-step physics — convection → large-scale condensation → grey radiation →
surface fluxes → boundary-layer diffusion → slab ocean — is *identical* to the
full model. What the column drops is everything dynamical:

```{list-table}
:header-rows: 1
:widths: 34 33 33

* - Quantity
  - Full model
  - Single column
* - horizontal transport
  - spectral advection
  - none
* - gravity-wave / spectral damping
  - yes
  - none
* - winds $u, v$
  - prognostic
  - **prescribed, fixed**
* - surface pressure $p_s$
  - prognostic
  - **fixed**
* - temperature $T$
  - spectral + physics
  - **grid leapfrog of physics** $dt_T$
* - humidity $q$
  - tracer advection + physics
  - **grid leapfrog of physics** $dt_q$
* - slab SST `t_surf`
  - mixed layer
  - mixed layer
```

So one SCM step runs the physics on the previous time level, then leapfrogs
**only $T$ and $q$** forward with the physics tendencies. The momentum tendency
from the boundary layer is computed and discarded — the SCM's stated assumption
is that "the dynamics" would restore the prescribed surface wind, so net
$du/dt = 0$. Surface pressure never changes.

## Usage

```python
from jsca.model import column as C

model  = C.build_column()          # 1 column, Frierson 25-level grid, global-avg lat
state0 = C.initial_state(model)    # cold start: T = 264 K, q = 1e-3, u_surf = 5 m/s

state, clim = C.integrate_climatology(
    model, state0, spinup_steps=..., avg_steps=..., cold_start=True,
)
```

`build_column` also runs a set of **independent** columns via `latitudes` /
`longitudes` — each responds to its own insolation, with no horizontal coupling.

## Physics options validated here

The SCM is the fast bench for swappable physics; each option is opt-in (default =
the Frierson config) and validated against Isca in the column before it runs in
the full 3D model:

- **`rad_scheme`** — grey longwave: `'frierson'` or `'byrne'`.
- **`do_seasonal`** — seasonal + diurnal insolation via the ported `astronomy`
  module (`solday ≥ 0` freezes the season with a diurnal cycle).
- **`convection_scheme`** — `SIMPLE_BETTS_MILLER`, `FULL_BETTS_MILLER`, `DRY`, or
  `NONE`.

## Validation

Against a real Isca column run on Isca's own 31 even-sigma levels, same
latitude / timestep / cold-start IC and the canonical `column_test.py` physics,
40-day agreement is essentially machine-adjacent:

```{list-table}
:header-rows: 1
:widths: 50 50

* - Diagnostic
  - Agreement (jsca vs Isca)
* - day-40 SST
  - Δ 0.007 K
* - day-40 precip
  - Δ 0.005 mm/day
* - day-40 $T$ profile
  - RMSD 0.13 K
* - day-40 $q$ profile
  - RMSD 0.11 g/kg
* - SST trajectory (40 d)
  - RMSD 0.012 K
```

Getting here drove out several subtle fidelity issues — Isca's deliberately
unstable slab initialization (`t_surf = init_temp + 1 K`), the
`do_lcl_diffusivity_depth` boundary-layer top, and the virtual-temperature
surface-flux stability. Each is documented on the relevant physics page. The full
account, the seasonal validation, and the latitude sweep are in
{doc}`../validation/index`.
