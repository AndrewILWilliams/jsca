# Vertical coordinate & pressure

`jsca` uses the general GFDL **hybrid σ/pressure vertical coordinate**, with
full-level pressures placed by the Simmons–Burridge (1981) scheme.

:::{admonition} Source
:class: note
Pressure & geopotential: {py:mod}`jsca.dycore.press_and_geopot` (ports
`press_and_geopot.F90`). Coordinate coefficients:
{py:mod}`jsca.dycore.vert_coordinate` (ports `vert_coordinate.F90`,
`compute_vert_coord`).
:::

## The hybrid coordinate

The vertical grid is defined by $K+1$ interface coefficient pairs $(a_k, b_k)$
(stored as `pk`, `bk`). Half-level (interface) pressures are

$$
p_{k+\frac12} = a_k + b_k\,p_s.
$$

Setting $a_k = 0$ recovers a pure sigma coordinate $p = \sigma p_s$; nonzero
$a_k$ blends toward pure pressure levels aloft.

### Coordinate options

```{list-table}
:header-rows: 1
:widths: 22 78

* - `vert_coord_option`
  - Definition
* - **`even_sigma`**
  - Isca's default: $a = 0$, $b_k = (k-1)/K$ — uniform σ, so
    $p_{\text{half}} = \sigma\,p_s$.
* - **`uneven_sigma`**
  - Stretched σ with enhanced surface resolution:
    $b = e^{-z/H}$, $z = \text{surf\_res}\,\zeta + (1-\text{surf\_res})\zeta^{\,\text{exponent}}$,
    $\zeta = 1 - (k-1)/K$.
* - **`hybrid`**
  - σ near the surface blending to pure pressure aloft via a $\sin^2$
    transition.
* - **`mcm`, `v197`**
  - Hard-coded 14- and 18-level tables.
* - **`input`**
  - Coefficients passed explicitly. The **Frierson** aquaplanet uses this — the
    25-level `frierson_test_case` table.
```

## Simmons–Burridge full-level placement

Pressures and their logs are built in `pressure_variables`. For the default
`vert_difference_option='simmons_and_burridge'`, the full-level log-pressures
follow Simmons & Burridge (1981):

$$
\alpha_k = 1 - \frac{p_{k+\frac12}\,\big(\ln p_{k+\frac12} - \ln p_{k-\frac12}\big)}
                    {p_{k+\frac12} - p_{k-\frac12}},
\qquad
\ln p_k = \ln p_{k+\frac12} - \alpha_k.
$$

These $\alpha_k$ coefficients are exactly what appear as `x1` in the
pressure-gradient force and `x4` in the omega-alpha heating on the
{doc}`primitive_equations` page — the discretization is internally consistent by
construction.

A zero model-top pressure (`pk[0] = bk[0] = 0`) is special-cased: the top
half-level log-pressure is set to a sentinel and
$\ln p_0 = \ln p_{\frac12} - 1$ (the hard-coded `ln_top_level_factor = -1`,
reproducing the Fortran behaviour for a $p_{\text{top}} = 0$ grid). The
alternative `mcm` option instead uses arithmetic-mean full levels
$p_k = \tfrac12(p_{k+\frac12} + p_{k-\frac12})$.

## Heights

Geopotential heights follow from the hydrostatic geopotential (see
{doc}`primitive_equations`) divided by $g$:
`compute_pressures_and_heights`, `compute_z_bot`.
