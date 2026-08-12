# API reference

Auto-generated documentation for the `jsca` package, built from the module and
function docstrings. The docstrings are the canonical statement of what each
routine ports and to what tolerance — they cite the Isca Fortran source (file and
lines) for every routine.

:::{admonition} Reading the modules
:class: tip
The package is organized as: {py:mod}`jsca.grid` (Gaussian grid + spectral
transforms), {py:mod}`jsca.dycore` (the spectral dynamical core),
{py:mod}`jsca.physics` (the column parameterizations), {py:mod}`jsca.model` (the
assembled configurations), {py:mod}`jsca.testing` (the equivalence statistics),
and {py:mod}`jsca.constants`.
:::

## Grid & spectral transforms

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   jsca.constants
   jsca.grid.gaussian
   jsca.grid.legendre
   jsca.grid.spectral
   jsca.grid.transforms
```

## Dynamical core

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   jsca.dycore.dynamics
   jsca.dycore.spectral_dynamics
   jsca.dycore.leapfrog
   jsca.dycore.implicit
   jsca.dycore.matrix_invert
   jsca.dycore.press_and_geopot
   jsca.dycore.spectral_damping
   jsca.dycore.vert_coordinate
   jsca.dycore.fv_advection
   jsca.dycore.vert_advection
   jsca.dycore.water_borrowing
   jsca.dycore.global_integral
```

## Physics parameterizations

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   jsca.physics.two_stream_gray_rad
   jsca.physics.astronomy
   jsca.physics.qe_moist_convection
   jsca.physics.betts_miller
   jsca.physics.dry_convection
   jsca.physics.lscale_cond
   jsca.physics.sat_vapor_pres
   jsca.physics.monin_obukhov
   jsca.physics.surface_flux
   jsca.physics.diffusivity
   jsca.physics.vert_diff
   jsca.physics.mixed_layer
   jsca.physics.bucket
   jsca.physics.damping_driver
   jsca.physics.hs_forcing
```

## Models

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   jsca.model.held_suarez
   jsca.model.frierson
   jsca.model.column
   jsca.model.bucket_model
   jsca.model.idealized_moist_phys
   jsca.model.land
```

## Testing utilities

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   jsca.testing.equivalence
```
