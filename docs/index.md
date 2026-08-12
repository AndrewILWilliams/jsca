---
sd_hide_title: true
---

# jsca

<div align="center">

# jsca

**A faithful JAX port of the [Isca](https://execlim.github.io/Isca/) idealized global climate model**

*The GFDL spectral dynamical core and the grey-radiation moist-aquaplanet
physics suite, re-expressed as pure, differentiable, `jit`-compilable JAX —
and validated against the Fortran, module by module and climate by climate.*

</div>

---

`jsca` reimplements Isca — the University of Exeter idealized GCM built on the
GFDL spectral core — in Python/JAX. The design goal is **fidelity, not
novelty**: every ported routine reproduces its Fortran ancestor's algorithm,
constants, and call order, and is pinned to golden fixtures dumped from the real
Fortran. A fast model that does not match Isca is considered a failure.

The payoff of the rewrite is what JAX buys for free once fidelity is proven:
`float64` end-to-end, `jit`/`vmap`/`scan`-composable time stepping, trivial
ensemble parallelism, GPU portability, and full differentiability of the model.

:::{admonition} Where things stand
:class: tip

The **full GFDL spectral dynamical core** and four complete model
configurations — dry **Held–Suarez**, the moist **Frierson aquaplanet**, the
**single-column model**, and a **land + bucket-hydrology** model — are ported and
statistically validated against pinned Isca runs. See {doc}`overview` for the
component-by-component status and {doc}`validation/index` for the climatology
comparisons.
:::

## Explore the documentation

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Getting started
:link: getting_started
:link-type: doc

Install `jsca`, run the test suite, and step a model forward.
:::

:::{grid-item-card} {octicon}`beaker` Governing equations
:link: governing_equations/index
:link-type: doc

The spectral primitive equations, the vertical coordinate, the spectral
transforms, and the semi-implicit leapfrog time integration.
:::

:::{grid-item-card} {octicon}`cloud` Physics parameterizations
:link: physics/index
:link-type: doc

Grey radiation, convection, large-scale condensation, the boundary layer,
surface fluxes, the slab ocean, and bucket hydrology.
:::

:::{grid-item-card} {octicon}`telescope` Model configurations
:link: models/index
:link-type: doc

Held–Suarez, Frierson, the single-column model, and the bucket-hydrology model.
:::

:::{grid-item-card} {octicon}`image` Example gallery
:link: gallery/index
:link-type: doc

Runnable scripts and the figures they produce — each validating a component
against Isca.
:::

:::{grid-item-card} {octicon}`check-circle` Validation
:link: validation/index
:link-type: doc

The end-to-end climatology comparisons: jsca vs pinned Isca.
:::

::::

## The iron rules

`jsca` is developed under a small set of non-negotiable rules that shape every
page of this manual:

1. **Faithful means faithful.** Port Isca's algorithm, constants, and call
   order; cite the Fortran source in the docstring; never silently "fix" or
   modernize Fortran behaviour — reproduce it and flag oddities.
2. **Every ported routine ships with Tier-1 fixtures from the real Fortran**,
   with tolerances from `rtol ≤ 1e-14` (pure arithmetic) down to `≤ 1e-11` for
   documented algorithm deviations.
3. **Fixtures come only from Fortran** — never generated or "refreshed" from
   Python output.
4. **JAX discipline:** `float64`, pure functions of `(config, params, state)`,
   no module-level state, everything on the step path `jit`/`vmap`/`scan`-safe.

```{toctree}
:hidden:
:caption: Getting started

getting_started
overview
```

```{toctree}
:hidden:
:caption: Reference
:maxdepth: 2

governing_equations/index
physics/index
models/index
```

```{toctree}
:hidden:
:caption: Examples & validation
:maxdepth: 2

gallery/index
validation/index
```

```{toctree}
:hidden:
:caption: API & bibliography

api/index
references
```
