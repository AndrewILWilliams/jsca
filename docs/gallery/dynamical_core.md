# Gallery — dynamical core

Examples validating the spectral core's transport and damping operators against
Isca. Return to the {doc}`index`.

(gallery-ppm)=
## PPM vertical advection

:::{admonition} `scripts/plot_ppm_advection.py` · fixture
:class: note
Reads `tests/fixtures/ppm_advection_reference.npz` (committed Fortran golden
data).
:::

Validates the GFDL finite-volume Piecewise-Parabolic-Method (PPM) vertical
advection operator ({py:mod}`jsca.dycore.fv_advection`) against Isca. The left
panel shows the vertical advective-tendency profile for a strong-wind column
where the vertical Courant number exceeds 1 (so the multi-cell departure-point
integral is active); the right panel is a jsca-vs-Isca scatter across both gentle
(Courant < 1) and strong regimes.

```{figure} ../figures/ppm_advection.png
:alt: PPM vertical advection, jsca vs Isca
jsca reproduces the Isca PPM tendency, including the Courant > 1 sub-stepping.
```

```{literalinclude} ../../scripts/plot_ppm_advection.py
:language: python
:caption: scripts/plot_ppm_advection.py
```

(gallery-update-tracers)=
## Grid tracer time-step

:::{admonition} `scripts/plot_update_tracers.py` · fixture
:class: note
Reads `tests/fixtures/update_tracers_reference.npz`.
:::

Validates the grid-space tracer time-step (`update_grid_tracer`): the
Robert–Asselin–Williams leapfrog filter plus vertical advection applied to
specific humidity — a core dycore stepping routine. The left panel shows, for one
column, the previous- and current-level humidity going in and the two outputs
(the RAW-filtered current level and the advected future level); the right panel
is a jsca-vs-Isca scatter of all three step outputs.

```{figure} ../figures/update_tracers.png
:alt: Grid tracer time-step, jsca vs Isca
```

```{literalinclude} ../../scripts/plot_update_tracers.py
:language: python
:caption: scripts/plot_update_tracers.py
```

(gallery-water-correction)=
## Global water conservation

:::{admonition} `scripts/plot_water_correction.py` · fixture
:class: note
Reads `tests/fixtures/compute_corrections_reference.npz`.
:::

Validates the global humidity-conservation correction — the MiMA-style
pressure-limited water rescaling that keeps total column water conserved after
spectral transport. The left panel plots the ratio of corrected-to-input
humidity against pressure: one constant rescale factor below the 200 hPa limit,
exactly 1 above it. The right panel is a jsca-vs-Isca scatter of corrected
humidity at every grid point.

```{figure} ../figures/water_correction.png
:alt: Water conservation correction, jsca vs Isca
```

```{literalinclude} ../../scripts/plot_water_correction.py
:language: python
:caption: scripts/plot_water_correction.py
```

(gallery-damping)=
## Rayleigh sponge (`damping_driver`)

:::{admonition} `scripts/plot_damping.py` · fixture
:class: note
Reads `tests/fixtures/damping_reference.npz`.
:::

Validates `damping_driver`, the top-of-model Rayleigh sponge that absorbs
vertically propagating waves near the lid. The left panel shows the sponge
wind-tendency profile — zero through the troposphere, ramping sharply toward the
model lid below `sponge_pbottom`; the right panel is a jsca-vs-Isca scatter of
u/v drag and frictional heating.

```{figure} ../figures/damping.png
:alt: Rayleigh sponge tendencies, jsca vs Isca
```

```{literalinclude} ../../scripts/plot_damping.py
:language: python
:caption: scripts/plot_damping.py
```

## Related figures without a standalone script

A few early dynamical-core figures were rendered by throwaway code inside their
originating pull requests and have no committed generating script. The components
they validate are covered by the fixture tests noted below:

- **Surface pressure-gradient force** (`pr11_pressure_gradient.png`) —
  `tests/test_pressure_gradient.py`.
- **Wind ↔ vorticity/divergence transforms** (`pr12_wind_vordiv.png`) —
  `tests/test_wind_transforms.py`.
- **Assembled dry core tendencies** (`pr13_dynamics_core.png`) —
  `tests/test_dynamics.py`.
- **Triangular-truncation stability fix** (`hs_triangular_truncation_fix.png`) —
  the T42, dt = 600 s blow-up fix described in
  {doc}`../governing_equations/spectral_representation`.
```
