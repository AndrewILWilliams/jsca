# Model configurations

`jsca` assembles the dynamical core and physics into four complete, validated
model configurations. They share the same anatomy — a `build_*` constructor
returning a static model object, an `initial_state`, and `step` /
`integrate` / `integrate_climatology` functions — and differ in which physics
they switch on and which prognostics they step.

```{list-table}
:header-rows: 1
:widths: 24 20 56

* - Configuration
  - Module
  - What it is
* - {doc}`held_suarez`
  - {py:mod}`jsca.model.held_suarez`
  - Dry dynamical core + Held–Suarez Newtonian/Rayleigh forcing. The
    dry-benchmark fidelity gate.
* - {doc}`frierson`
  - {py:mod}`jsca.model.frierson`
  - Moist aquaplanet: grey radiation, Betts–Miller convection, large-scale
    condensation, boundary layer, slab ocean.
* - {doc}`column`
  - {py:mod}`jsca.model.column`
  - Single-column model — the full column physics with the dynamical core
    bypassed. The fast physics-validation bench.
* - {doc}`bucket`
  - {py:mod}`jsca.model.bucket_model`
  - The Frierson stack over a land mask with a Manabe soil-moisture bucket.
```

The intermediate driver {py:mod}`jsca.model.idealized_moist_phys` assembles the
ten column-physics modules in Isca's exact call order (see {doc}`../physics/index`)
and is shared by the Frierson, column, and bucket models.

```{toctree}
:hidden:
:maxdepth: 1

held_suarez
frierson
column
bucket
```

## The common pattern

```python
import jsca
from jsca.model import frierson as F

model  = F.build_frierson(num_fourier=42)   # static config + precomputed params
state0 = F.initial_state(model)             # a prognostic pytree

# Step, or spin up and time-average.
state, clim = F.integrate_climatology(
    model, state0, spinup_steps=..., avg_steps=..., cold_start=True,
)
```

`build_*` returns a frozen model object bundling the config, the precomputed
`params` (Gaussian weights, Legendre tables, damping coefficients, the
semi-implicit wave matrices, …), and the grid. `step` is a pure function; the
whole loop composes with `lax.scan`. See {doc}`../validation/index` for how each
configuration is checked against pinned Isca.
