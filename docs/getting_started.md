# Getting started

## Installation

`jsca` is a pure-Python package. Its only runtime dependencies are JAX, NumPy,
and SciPy.

```bash
git clone https://github.com/AndrewILWilliams/jsca
cd jsca
pip install -e .[dev]      # dev adds pytest, ruff, mypy
```

`float64` is enabled the moment you `import jsca` (it calls
`jax.config.update("jax_enable_x64", True)`), so every array in the model is
double precision by default — a requirement for matching the Fortran, which is
compiled with `-fdefault-real-8`.

Optional extras:

```bash
pip install -e .[io]       # xarray + netcdf4 + cftime, for reading Isca output
```

## Run the test suite

The fixture suite is the contract that keeps `jsca` faithful to Isca. Every
ported routine is checked against golden input/output data dumped from the real
Fortran. Run it first:

```bash
pytest                     # the full grid / dycore / physics fixture suite
ruff check src tests bench # lint gate (must pass before any push)
```

The tolerances are tight by design: `rtol ≤ 1e-14` for pure arithmetic,
`≤ 1e-13` for log/exp-bearing kernels, and `≤ 1e-11` for the handful of
documented algorithm deviations (e.g. the LAPACK matrix inverse). If `pytest`
passes, the port matches the Fortran to those tolerances.

## Step a model forward

Every model configuration follows the same shape: build a model object (static
config + precomputed numeric tables), construct an initial state, then step.
Here is the single-column model — the fastest way to see the physics run:

```python
import jsca
from jsca.model import column as C

model  = C.build_column()          # 1 column, Frierson 25-level grid
state0 = C.initial_state(model)    # cold start: T = 264 K, q = 1e-3, u_surf = 5 m/s

# Spin up and return a time-mean climatology.
state, clim = C.integrate_climatology(
    model, state0, spinup_steps=2000, avg_steps=2000, cold_start=True,
)
print(clim["t_surf"], clim["precip"])   # (nlat, nlon) surface T and precip
```

The moist aquaplanet is built the same way from
{py:mod}`jsca.model.frierson`, the dry benchmark from
{py:mod}`jsca.model.held_suarez`, and the land run from
{py:mod}`jsca.model.bucket_model`. See {doc}`models/index` for each.

## The pinned Fortran reference

Every port and fixture in `jsca` refers to one pinned Isca commit:

```bash
git clone https://github.com/ExeClim/Isca /tmp/isca && cd /tmp/isca
git checkout a290bc376d84d0ee83adbb80eb374b9f629c3534   # the validation target
export ISCA_SRC=/tmp/isca
```

You do **not** need Isca or a Fortran compiler to use `jsca` or to run its CI:
the golden fixtures and reference climatologies are committed to the repository.
Fortran is needed only to *regenerate* fixtures (see the fixture workflow in
`CLAUDE.md`).

## How this manual is organized

- {doc}`governing_equations/index` — the equations the dynamical core solves and
  the numerics it uses.
- {doc}`physics/index` — each column-physics parameterization, its equations, its
  paper, and the Isca source it ports.
- {doc}`models/index` — how the pieces assemble into the four model
  configurations.
- {doc}`gallery/index` — runnable example scripts and the figures they produce.
- {doc}`validation/index` — the climatology comparisons against pinned Isca.
