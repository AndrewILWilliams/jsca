# Gallery — convection & condensation

Examples validating the convection and condensation schemes against Isca. Return
to the {doc}`index`. The physics is described in {doc}`../physics/convection`.

(gallery-qe-cape)=
## CAPE / CIN diagnosis (simplified Betts–Miller)

:::{admonition} `scripts/plot_qe_moist_convection.py` · fixture
:class: note
Reads `tests/fixtures/qe_moist_convection_reference.npz`.
:::

Validates the CAPE/CIN calculation stage of the simplified (quasi-equilibrium)
Betts–Miller scheme — the parcel-lifting instability diagnosis. The left panel
shows environment vs lifted-parcel virtual temperature with CAPE (parcel warmer)
and CIN (parcel cooler) areas shaded; the right panel is a jsca-vs-Isca scatter
of CAPE and CIN across all columns.

```{figure} ../figures/qe_moist_convection.png
:alt: CAPE/CIN diagnosis, jsca vs Isca
```

```{literalinclude} ../../scripts/plot_qe_moist_convection.py
:language: python
:caption: scripts/plot_qe_moist_convection.py
```

(gallery-qe-adjust)=
## Betts–Miller adjustment

:::{admonition} `scripts/plot_qe_moist_convection_adjust.py` · fixture
:class: note
Reads `tests/fixtures/qe_moist_convection_reference.npz`.
:::

Validates the adjustment/relaxation stage of the simplified Betts–Miller scheme —
relaxation of $T$ and $q$ toward reference profiles and the resulting convective
precipitation. The left panel shows the temperature and humidity increments
applied to a deep-convecting column (jsca lines over Isca markers): heating aloft,
drying.

```{figure} ../figures/qe_moist_convection_adjust.png
:alt: Betts-Miller adjustment, jsca vs Isca
```

```{literalinclude} ../../scripts/plot_qe_moist_convection_adjust.py
:language: python
:caption: scripts/plot_qe_moist_convection_adjust.py
```

(gallery-full-bm)=
## Full Betts–Miller

:::{admonition} `scripts/plot_betts_miller.py` · fixture
:class: note
Reads `tests/fixtures/betts_miller_reference.npz`.
:::

Validates the full Betts–Miller scheme (`convection_scheme='FULL_BETTS_MILLER'`,
`betts_miller.f90`) — distinct from the simplified qe scheme, with its own
`capecalcnew` and hardcoded LCL lookup table. The left panel shows the
temperature and humidity increments for a deep-convecting column (jsca over
Isca); the right panel is a jsca-vs-Isca scatter of the temperature increment
across all columns and levels.

```{figure} ../figures/betts_miller.png
:alt: Full Betts-Miller, jsca vs Isca
```

```{literalinclude} ../../scripts/plot_betts_miller.py
:language: python
:caption: scripts/plot_betts_miller.py
```

(gallery-lscale)=
## Large-scale condensation

:::{admonition} `scripts/plot_lscale_cond.py` · fixture
:class: note
Reads `tests/fixtures/lscale_cond_reference.npz`.
:::

Validates large-scale (stratiform) condensation `lscale_cond` — the grid-scale
removal of supersaturation and optional re-evaporation of falling precipitation.
The left panel shows a column's input humidity vs saturation and the resulting
$T/q$ increments (condensation removing supersaturation, re-evaporation moistening
a dry layer below).

```{figure} ../figures/lscale_cond.png
:alt: Large-scale condensation, jsca vs Isca
```

```{literalinclude} ../../scripts/plot_lscale_cond.py
:language: python
:caption: scripts/plot_lscale_cond.py
```

(gallery-dry-conv)=
## Dry convection (Schneider–Walker)

:::{admonition} `scripts/plot_dry_convection.py` · fixture
:class: note
Reads `tests/fixtures/dry_convection_reference.npz`.
:::

Validates the Schneider–Walker dry convective adjustment
(`convection_scheme='DRY'`) — relaxation of super-adiabatic layers toward a
prescribed lapse rate, conserving column enthalpy. The left panel shows the
environment temperature, the energy-conserving adjusted profile, and the
tendency for a convecting column (jsca over Isca).

```{figure} ../figures/dry_convection.png
:alt: Dry convection, jsca vs Isca
```

```{literalinclude} ../../scripts/plot_dry_convection.py
:language: python
:caption: scripts/plot_dry_convection.py
```
