# Physical constants

`jsca`'s constants are copied **verbatim** from Isca's `constants.F90`. Because
Isca compiles with `-fdefault-real-8`, all are `float64`. Reproducing Isca means
reproducing its constants — including the deliberately non-textbook choices. The
module docstring is explicit: **do not "correct" values that look off.**

:::{admonition} Source
:class: note
{py:mod}`jsca.constants` (ports `shared/constants/constants.F90`).
:::

```{list-table} Key constants (Isca's non-textbook choices flagged)
:header-rows: 1
:widths: 22 30 48

* - Constant
  - jsca value
  - Note
* - `RADIUS` ($a$)
  - **6376.0 × 10³ m**
  - **Non-textbook** — Isca default, *not* 6371 km. Enters $a^2$ in every
    Laplacian/eigenvalue and metric term.
* - `GRAV` ($g$)
  - **9.80 m s⁻²**
  - **Non-textbook** — Isca's value, *not* 9.81.
* - `OMEGA` ($\Omega$)
  - 7.2921150 × 10⁻⁵ s⁻¹
  - Earth rotation rate.
* - `RDGAS` ($R_d$)
  - 287.04 J kg⁻¹ K⁻¹
  - Dry-air gas constant.
* - `KAPPA` ($\kappa$)
  - 2/7
  - $= R_d/c_p$.
* - `CP_AIR` ($c_p$)
  - 1004.64 J kg⁻¹ K⁻¹
  - Derived as $R_d/\kappa$, so exactly consistent with $\kappa$.
* - `RVGAS` ($R_v$)
  - 461.50 J kg⁻¹ K⁻¹
  - Water-vapour gas constant.
* - `HLV`
  - 2.500 × 10⁶ J kg⁻¹
  - Latent heat of vaporization.
* - `HLF`
  - 3.34 × 10⁵ J kg⁻¹
  - Latent heat of fusion.
* - `STEFAN` ($\sigma$)
  - **5.6734 × 10⁻⁸ W m⁻² K⁻⁴**
  - **Non-textbook** — Isca's value (the standard is 5.670374 × 10⁻⁸).
* - `PSTD_MKS`
  - 101325.0 Pa
  - Reference sea-level pressure.
* - `CP_OCEAN`
  - 3989.24495292815 J kg⁻¹ K⁻¹
  - Slab-ocean heat capacity.
* - `RHO0`
  - 1.035 × 10³ kg m⁻³
  - Ocean reference density.
* - `VONKARM`
  - 0.40
  - von Kármán constant.
* - `TFREEZE`
  - 273.16 K
  - (Note: `KELVIN = 273.15` is a separate constant.)
* - `GAS_CONSTANT`
  - **8.314 J mol⁻¹ K⁻¹**
  - **Non-textbook** — Isca's value.
* - `SECONDS_PER_DAY`
  - 86400.0
  -
```

Several planetary constants (`RADIUS`, `OMEGA`, `GRAV`, `RDGAS`, `KAPPA`,
`PSTD_MKS`, …) are `real, public ::` variables in Isca — runtime-configurable via
`constants_nml` — rather than `parameter`s; `jsca` exposes them as config
defaults. The most physically visible consequences of the non-textbook choices
are the 6376 km radius (in every Laplacian eigenvalue and metric term) and
$g = 9.80$ (in the hydrostatic $\Phi/g$ and the mass weighting $\Delta p/g$).
