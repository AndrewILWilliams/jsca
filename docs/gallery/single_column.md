# Gallery — single-column model

Examples validating the full column-physics stack (dynamical core bypassed)
against Isca. Return to the {doc}`index`. The model is described in
{doc}`../models/column`.

(gallery-scm-spinup)=
## SCM spin-up to radiative–convective equilibrium

:::{admonition} `scripts/plot_column_scm.py` · pure
:class: note
No Isca reference — a pure jsca behaviour demonstration.
:::

Shows jsca's single-column model spinning up from Isca's cold start to
radiative–convective equilibrium (~400 days) — a stability/behaviour
demonstration that the physics-chain assembly is stable and settles into a
sensible tropical single-column climate under Frierson radiation and simplified
Betts–Miller convection.

```{figure} ../figures/column_scm_spinup.png
:alt: SCM spin-up to RCE
```

```{literalinclude} ../../scripts/plot_column_scm.py
:language: python
:caption: scripts/plot_column_scm.py
```

(gallery-scm-vs-isca)=
## SCM vs Isca (apples-to-apples)

:::{admonition} `scripts/compare_column_scm.py` · needs-Isca
:class: note
Takes an Isca `atmos_daily.nc`; a committed one exists at
`baseline/reference/column_scm_isca_daily_t264.nc`. Requires `xarray` + `netCDF4`.
:::

The Tier-2 physics-chain validation: the whole column-physics stack stepped
identically to Isca, matching to essentially machine-adjacent tolerances. jsca
reads Isca's actual `pk`/`bk`, latitude, timestep, and initial condition, then
overplots the spin-up trajectories and equilibrium profiles.

```{figure} ../figures/column_scm_vs_isca.png
:alt: SCM vs Isca
day-40 SST RMSD 0.012 K, T profile 0.13 K, q profile 0.11 g/kg.
```

```{literalinclude} ../../scripts/compare_column_scm.py
:language: python
:caption: scripts/compare_column_scm.py
```

(gallery-scm-seasonal)=
## Seasonal + diurnal insolation

:::{admonition} `scripts/plot_column_seasonal.py` · ref-committed
:class: note
Reads `baseline/reference/column_scm_isca_seasonal.npz` plus a live jsca run.
:::

Validates the `do_seasonal` seasonal + diurnal insolation path (via the ported
`astronomy` `diurnal_solar`) coupled through the SCM to the slab ocean, over a
90-day seasonal march. The panels show the daily-mean TOA insolation climbing from
NH winter toward summer (jsca over Isca), the slab SST trajectory responding to
it, and the day-90 temperature profile.

```{figure} ../figures/column_scm_seasonal_vs_isca.png
:alt: Seasonal SCM vs Isca
SST trajectory RMSD 0.008 K; interior days match to machine precision.
```

```{literalinclude} ../../scripts/plot_column_seasonal.py
:language: python
:caption: scripts/plot_column_seasonal.py
```

(gallery-scm-sweep)=
## Latitude sweep

:::{admonition} `scripts/plot_column_sweep.py` · ref-committed
:class: note
Reads `baseline/reference/column_scm_isca_sweep.npz` plus live jsca runs.
:::

Confirms the column physics reproduce Isca's meridional climate structure across
insolation regimes. The panels show the equator→pole structure both models
produce: final SST and precip vs latitude, and day-40 $T/q$ profiles at a tropical
(0°) and high-latitude (60°) column.

```{figure} ../figures/column_scm_sweep_vs_isca.png
:alt: SCM latitude sweep vs Isca
```

```{literalinclude} ../../scripts/plot_column_sweep.py
:language: python
:caption: scripts/plot_column_sweep.py
```
