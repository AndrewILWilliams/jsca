# Time integration

`jsca` advances the spectral state with a **semi-implicit leapfrog** scheme: the
fast linear gravity-wave terms are treated implicitly (so the timestep is set by
advection, not gravity waves), and a **Robert–Asselin–Williams (RAW)** filter
controls the leapfrog computational mode.

:::{admonition} Source
:class: note
{py:mod}`jsca.dycore.leapfrog` (ports `leapfrog.F90`),
{py:mod}`jsca.dycore.implicit` (ports `implicit.F90`),
{py:mod}`jsca.dycore.matrix_invert` (ports `matrix_invert.F90`).
:::

## Leapfrog with the RAW filter

Three time levels are stored on the last array axis (`previous`, `current`,
`future`). With filter strength $\nu$ (`robert_coeff`) and Williams' parameter
$\alpha$ (`raw_filter_coeff`; $\alpha = 1$ recovers the classic Robert–Asselin
filter), the combined scheme (`leapfrog`) is

$$
P = X^{t-\Delta t} - 2X^{t},
$$
$$
X^{t+\Delta t} = X^{t-\Delta t} + 2\Delta t\,\frac{dX}{dt},
$$
$$
X^{t} \leftarrow X^{t} + \nu\,\alpha\,\big(P + X^{t+\Delta t}\big),
$$
$$
X^{t+\Delta t} \leftarrow X^{t+\Delta t} + \nu\,(\alpha - 1)\,\big(P + X^{t+\Delta t}\big).
$$

The Williams (2011) modification ($\alpha < 1$) applies the filter correction to
**both** the current and future levels, damping the computational mode while
preserving third-order accuracy — a genuine improvement over the classic filter,
which degrades the physical mode to first order.

### Two-level split

For the semi-implicit correction the filter is split around the implicit
adjustment: `leapfrog_2level_a` produces `(a_updated, P)`, the future value is
corrected by the semi-implicit solve, then `leapfrog_2level_b` completes the
filter using the threaded `P`. This lets the gravity-wave adjustment sit between
the two halves of the filter.

:::{admonition} Two preserved Fortran quirks (reproduced deliberately)
:class: caution
Rule 1 — *faithful means faithful* — means reproducing Isca's behaviour even
where it is idiosyncratic:

1. The grid-tracer variant `leapfrog_2level_a_real` **omits** the
   `raw_filter_coeff` factor on the current-level update (`leapfrog.F90` L128),
   whereas the complex variant applies it (L77).
2. In `update_grid_tracer` (F90 L1243 vs L1248) the future-level RAW correction
   is computed and then immediately overwritten by `q_future = tr` — so the
   tracer future level is simply the advected value. `jsca` reproduces this
   dead-store.
:::

## Semi-implicit treatment

The linear gravity-wave terms $\mathcal{L}$ — the divergence ↔ temperature ↔
surface-pressure coupling, linearized about a reference profile — are treated
implicitly:

$$
\frac{X^{t+\Delta t} - X^{t-\Delta t}}{2\Delta t}
  = \mathcal{N}(X^t)
  + \mathcal{L}\,\frac{(1+\epsilon)X^{t+\Delta t} + (1-\epsilon)X^{t-\Delta t}}{2},
$$

with $\xi = \alpha_{\text{imp}}\,\Delta t$ (`alpha_implicit`, default 0.5 →
centred / Crank–Nicolson). `implicit_init` builds the timestep-independent
divergence-coupling matrix and reference-pressure derivatives once
(`implicit.F90` L171–217); the linear operators are `_linear_tp_tendency`
(linearized $(T, \ln p_s)$ tendency from divergence), `_linear_geopotential`, and
`_pres_grad_funct`.

## The vertical matrix solve

The implicit divergence step reduces, per total wavenumber $l$, to a small dense
$K\times K$ vertical solve:

$$
\delta^{t+\Delta t}_{\text{corr}}
  = \left(I + \xi^2\,\frac{l(l+1)}{a^2}\,\text{div\_mat}\right)^{-1}\,
    \delta^{\text{explicit}}.
$$

`build_wave_matrices` stacks these inverses over $l = 0 \dots$
`num_total_wavenumbers`; `implicit_correction` (F90 L241–286) runs the per-$(m,n)$
matrix–vector solve, gathering the inverse for $l = m + n$, then applies the
consistent $T$ and $\ln p_s$ updates.

:::{admonition} Documented deviation — LAPACK for the matrix inverse
:class: note
{py:mod}`jsca.dycore.matrix_invert` replaces the Fortran's
Gauss–Jordan-with-pivoting loop with `jnp.linalg.inv` / `det` (LAPACK/XLA). The
result is mathematically identical and fixture-proven to a relative tolerance of
$\sim 10^{-11}$ for the diagonally-dominant, `nlev`-sized matrices — the one
place a library routine replaces the literal Fortran loop, as allowed by the
iron rules when documented with its tolerance.
:::
