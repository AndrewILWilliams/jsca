# Example gallery

Every example below pairs a **runnable script** with the **figure it produces**.
Most examples validate one `jsca` component against Isca: the script feeds
`jsca` the same inputs the Fortran saw and overplots the two, so the figure *is*
the fidelity check. Click any thumbnail to jump to its code and discussion.

:::{admonition} How to read the runnability tags
:class: tip

Each example is tagged by what data it needs, so you know what you can reproduce
offline:

- **pure** — pure `jsca`, no reference data. Runs from nothing.
- **fixture** — reads a *committed* Fortran fixture in `tests/fixtures/`. Runs
  standalone; no Fortran needed.
- **ref-committed** — reads *committed* Isca-derived reference data in
  `baseline/reference/`. Regenerates offline.
- **needs-Isca** — requires a fresh external Isca run / netCDF. The committed
  figure stands in; the reference is regenerated from an Isca run.
:::

## Dynamical core

::::{grid} 2 2 3 3
:gutter: 2

:::{grid-item-card}
:link: dynamical_core
:link-type: doc
:img-top: ../figures/ppm_advection.png
PPM vertical advection
:::
:::{grid-item-card}
:link: dynamical_core
:link-type: doc
:img-top: ../figures/update_tracers.png
Grid tracer time-step
:::
:::{grid-item-card}
:link: dynamical_core
:link-type: doc
:img-top: ../figures/water_correction.png
Water conservation
:::
:::{grid-item-card}
:link: dynamical_core
:link-type: doc
:img-top: ../figures/damping.png
Rayleigh sponge
:::
::::

## Radiation

::::{grid} 2 2 3 3
:gutter: 2

:::{grid-item-card}
:link: radiation
:link-type: doc
:img-top: ../figures/two_stream_gray_rad.png
Grey radiation (Frierson)
:::
:::{grid-item-card}
:link: radiation
:link-type: doc
:img-top: ../figures/two_stream_gray_rad_byrne.png
Grey radiation (Byrne)
:::
::::

## Convection & condensation

::::{grid} 2 2 3 3
:gutter: 2

:::{grid-item-card}
:link: convection
:link-type: doc
:img-top: ../figures/qe_moist_convection.png
CAPE / CIN diagnosis
:::
:::{grid-item-card}
:link: convection
:link-type: doc
:img-top: ../figures/qe_moist_convection_adjust.png
Betts–Miller adjustment
:::
:::{grid-item-card}
:link: convection
:link-type: doc
:img-top: ../figures/betts_miller.png
Full Betts–Miller
:::
:::{grid-item-card}
:link: convection
:link-type: doc
:img-top: ../figures/lscale_cond.png
Large-scale condensation
:::
:::{grid-item-card}
:link: convection
:link-type: doc
:img-top: ../figures/dry_convection.png
Dry convection
:::
::::

## Boundary layer & surface

::::{grid} 2 2 3 3
:gutter: 2

:::{grid-item-card}
:link: boundary_layer
:link-type: doc
:img-top: ../figures/monin_obukhov.png
Monin–Obukhov drag
:::
:::{grid-item-card}
:link: boundary_layer
:link-type: doc
:img-top: ../figures/surface_flux.png
Bulk surface fluxes
:::
:::{grid-item-card}
:link: boundary_layer
:link-type: doc
:img-top: ../figures/diffusivity.png
PBL diffusivity
:::
:::{grid-item-card}
:link: boundary_layer
:link-type: doc
:img-top: ../figures/vert_diff.png
Vertical diffusion
:::
::::

## Mixed layer, land & bucket

::::{grid} 2 2 3 3
:gutter: 2

:::{grid-item-card}
:link: surface_land
:link-type: doc
:img-top: ../figures/mixed_layer.png
Slab-ocean mixed layer
:::
:::{grid-item-card}
:link: surface_land
:link-type: doc
:img-top: ../figures/bucket_stepping.png
Bucket hydrology
:::
::::

## Full-model climatology

::::{grid} 2 2 3 3
:gutter: 2

:::{grid-item-card}
:link: climatology
:link-type: doc
:img-top: ../figures/hs_jsca_vs_isca_dt600.png
Held–Suarez vs Isca
:::
:::{grid-item-card}
:link: climatology
:link-type: doc
:img-top: ../figures/frierson_climatology_matched.png
Frierson vs Isca
:::
:::{grid-item-card}
:link: climatology
:link-type: doc
:img-top: ../figures/frierson_ensemble_parity.png
Statistical parity
:::
:::{grid-item-card}
:link: climatology
:link-type: doc
:img-top: ../figures/frierson_solar_response.png
+2 % solar response
:::
::::

## Single-column model

::::{grid} 2 2 3 3
:gutter: 2

:::{grid-item-card}
:link: single_column
:link-type: doc
:img-top: ../figures/column_scm_vs_isca.png
SCM vs Isca
:::
:::{grid-item-card}
:link: single_column
:link-type: doc
:img-top: ../figures/column_scm_seasonal_vs_isca.png
Seasonal insolation
:::
:::{grid-item-card}
:link: single_column
:link-type: doc
:img-top: ../figures/column_scm_sweep_vs_isca.png
Latitude sweep
:::
::::

```{toctree}
:hidden:

dynamical_core
radiation
convection
boundary_layer
surface_land
climatology
single_column
```
