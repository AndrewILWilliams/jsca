# Gallery — full-model climatology

The end-to-end tests: whole models integrated to statistical equilibrium and
compared against pinned Isca. Return to the {doc}`index`. The results are
discussed in {doc}`../validation/index`.

(gallery-hs)=
## Held–Suarez vs Isca

:::{admonition} `baseline/compare_hs.py` · ref-committed
:class: note
Reads the committed ensembles `baseline/reference/hs_isca_members.npz` and
`hs_jsca_members_dt600.npz`.
:::

A head-to-head Held–Suarez climatology validation of jsca's assembled dry
dynamical core against Isca at dt = 600 s, using ensemble-equivalence statistics.
The 2×3 panel shows Isca | jsca | difference for zonal-mean $u$ and $T$, plus a
Tier-3 equivalence test at each (level, latitude) point — pale everywhere, with
no significance stippling.

```{figure} ../figures/hs_jsca_vs_isca_dt600.png
:alt: Held-Suarez climatology, jsca vs Isca
0.0 % of points differ from Isca beyond its internal variability.
```

```{literalinclude} ../../baseline/compare_hs.py
:language: python
:caption: baseline/compare_hs.py
```

(gallery-frierson-clim)=
## Frierson aquaplanet vs Isca (like-for-like)

:::{admonition} `scripts/compare_frierson_climatology.py` · ref-committed
:class: note
Reads committed zonal-mean references in `baseline/reference/`. Run
`python scripts/compare_frierson_climatology.py matched` for the figure below.
:::

The strictest like-for-like moist-model comparison: jsca run on Isca's exact
64×128 grid with Isca's matched initial condition (days 200–300 mean), so nothing
but the code differs. Jets, temperature, humidity, and precipitation all sit
essentially on top of Isca's.

```{figure} ../figures/frierson_climatology_matched.png
:alt: Frierson climatology, jsca vs Isca, matched
Zonal-mean u, T, q, precip and t_surf: jsca on top of Isca.
```

```{literalinclude} ../../scripts/compare_frierson_climatology.py
:language: python
:caption: scripts/compare_frierson_climatology.py
```

(gallery-parity)=
## Statistical parity (Tier-3 equivalence)

:::{admonition} `scripts/compare_frierson_ensemble.py` · ref-committed
:class: note
Reads `frierson_isca_members_t21.npz` and `frierson_jsca_members_t21.npz`.
:::

The rigorous "within-sampling-parity" verdict for the Frierson climatology: at
each (level, latitude) point, whether jsca's 8-member ensemble mean lies within
Isca's own month-to-month internal variability (FDR-controlled). This is
equivalence testing rather than eyeballing means — the closer of issue #27.

```{figure} ../figures/frierson_ensemble_parity.png
:alt: Frierson ensemble parity map
```

```{literalinclude} ../../scripts/compare_frierson_ensemble.py
:language: python
:caption: scripts/compare_frierson_ensemble.py
```

(gallery-solar)=
## +2 % solar-constant response

:::{admonition} `scripts/compare_frierson_solar_response.py` · ref-committed
:class: note
Reads the control and perturbed member ensembles in `baseline/reference/`.
:::

A climate-*change* test: raise the solar constant by 2 % in both models, run to
equilibrium, and compare the *response* (perturbed − control). It probes whether
jsca reproduces Isca's *sensitivity*, not just its base climate — global-mean
warming, the wet-get-wetter precipitation signature, and tropical
upper-tropospheric amplification.

```{figure} ../figures/frierson_solar_response.png
:alt: +2% solar response, jsca vs Isca
```

```{literalinclude} ../../scripts/compare_frierson_solar_response.py
:language: python
:caption: scripts/compare_frierson_solar_response.py
```

(gallery-spinup)=
## Spin-up trajectories (global means)

:::{admonition} `scripts/plot_frierson_evolution.py` · ref-committed
:class: note
Reads the committed jsca/Isca evolution `.npz` files in `baseline/reference/`.
:::

Time series of area-weighted global means (mass-weighted $T$, precip, column
humidity, `t_surf`) for jsca vs real Isca from the same initial condition. This
is the diagnostic behind the water-conservation fix — it showed precip failing to
spin up for ~45 days when the evaporation source was being deleted each step.

```{figure} ../figures/frierson_t21_year_evolution.png
:alt: Frierson global-mean spin-up, jsca vs Isca
A full-year T21 run: multi-season stability and drift-free agreement.
```

```{literalinclude} ../../scripts/plot_frierson_evolution.py
:language: python
:caption: scripts/plot_frierson_evolution.py
```

(gallery-hs-pure)=
## jsca's own Held–Suarez (from rest)

:::{admonition} `bench/run_held_suarez.py` · pure
:class: note
No external input — a pure jsca run from an isothermal resting state (T21L15,
dt = 600 s, ~10 min CPU).
:::

Demonstrates that the full assembled dynamical core plus HS forcing produces the
expected HS94 general circulation — eddy-driven midlatitude jets and tropopause
structure — from rest, with no reference data at all.

```{figure} ../figures/pr14_held_suarez.png
:alt: jsca Held-Suarez climatology from rest
```

```{literalinclude} ../../bench/run_held_suarez.py
:language: python
:caption: bench/run_held_suarez.py
```
