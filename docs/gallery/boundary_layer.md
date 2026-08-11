# Gallery — boundary layer & surface

Examples validating the surface-layer, surface-flux, and vertical-mixing schemes
against Isca. Return to the {doc}`index`. The physics is described in
{doc}`../physics/boundary_layer`.

(gallery-mo)=
## Monin–Obukhov drag

:::{admonition} `scripts/plot_monin_obukhov.py` · fixture
:class: note
Reads `tests/fixtures/monin_obukhov_reference.npz`.
:::

Validates the Monin–Obukhov surface-layer similarity solver — stability-dependent
exchange coefficients from the bulk Richardson number. The left panel shows the
momentum drag coefficient vs bulk Richardson number: enhanced in unstable air,
collapsing to `drag_min` toward the critical Richardson number in stable air.

```{figure} ../figures/monin_obukhov.png
:alt: Monin-Obukhov drag, jsca vs Isca
```

```{literalinclude} ../../scripts/plot_monin_obukhov.py
:language: python
:caption: scripts/plot_monin_obukhov.py
```

(gallery-surface-flux)=
## Bulk surface fluxes

:::{admonition} `scripts/plot_surface_flux.py` · fixture
:class: note
Reads `tests/fixtures/surface_flux_reference.npz`.
:::

Validates the bulk aerodynamic surface-flux computation (`surface_flux`) over
ocean — turbulent sensible heat, evaporation, and wind stress from the
similarity exchange coefficients. The left panel shows the sensible-heat and
evaporative fluxes vs the air–sea temperature contrast (jsca over Isca).

```{figure} ../figures/surface_flux.png
:alt: Bulk surface fluxes, jsca vs Isca
```

```{literalinclude} ../../scripts/plot_surface_flux.py
:language: python
:caption: scripts/plot_surface_flux.py
```

(gallery-diffusivity)=
## PBL eddy diffusivity

:::{admonition} `scripts/plot_diffusivity.py` · fixture
:class: note
Reads `tests/fixtures/diffusivity_reference.npz` (the `do_simple=.false.` variant
is in `diffusivity_nosimple_reference.npz`).
:::

Validates the boundary-layer eddy-diffusivity $K$-profile scheme and PBL-height
diagnosis (`diffusivity`), including Isca's default virtual-temperature
parcel-buoyancy PBL. The panels show the momentum and heat eddy-diffusivity
profiles for a couple of columns (jsca over Isca) with the diagnosed PBL top
marked.

```{figure} ../figures/diffusivity.png
:alt: PBL diffusivity, jsca vs Isca
```

```{literalinclude} ../../scripts/plot_diffusivity.py
:language: python
:caption: scripts/plot_diffusivity.py
```

(gallery-vert-diff)=
## Implicit vertical diffusion

:::{admonition} `scripts/plot_vert_diff.py` · fixture
:class: note
Reads the `vert_diff` fixtures in `tests/fixtures/`.
:::

Validates the implicit vertical-diffusion solver (`vert_diff`) — the tridiagonal
implicit update that mixes momentum, heat, and moisture through the PBL using the
$K$-profile. The left panel shows the momentum and temperature diffusion
tendencies for one column (jsca over Isca), concentrated in the boundary layer.

```{figure} ../figures/vert_diff.png
:alt: Vertical diffusion, jsca vs Isca
```

```{literalinclude} ../../scripts/plot_vert_diff.py
:language: python
:caption: scripts/plot_vert_diff.py
```
