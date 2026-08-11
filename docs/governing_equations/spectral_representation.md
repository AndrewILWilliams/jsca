# Spectral representation & transforms

`jsca` represents horizontal fields as truncated **spherical-harmonic
expansions**, and moves between grid and spectral space with the classic
transform pair: a fast Fourier transform in longitude and a Gaussian-quadrature
Legendre transform in latitude.

:::{admonition} Source
:class: note
{py:mod}`jsca.grid.gaussian`, {py:mod}`jsca.grid.legendre`,
{py:mod}`jsca.grid.spectral`, {py:mod}`jsca.grid.transforms` — porting
`gauss_and_legendre.F90`, `spherical.F90`, `transforms.F90`.
:::

## Spherical-harmonic expansion

A field is expanded as

$$
X(\lambda_i, \mu_j) = \sum_{m=-M}^{M}\sum_{l}
    X_l^m\,\bar P_l^m(\mu_j)\,e^{\,i m \lambda_i},
\qquad
X_l^{-m} = \overline{X_l^{m}},
$$

where $\mu = \sin\phi$. Because the modelled fields are real, only $m \ge 0$ is
stored.

## GFDL normalization

The associated Legendre functions are **unit-integral normalized** with **no
Condon–Shortley phase**:

$$
\int_{-1}^{1}\big[\bar P_l^m(\mu)\big]^2\,d\mu = 1,
\qquad
\bar P_0^0 = \sqrt{\tfrac12}.
$$

(The relation to the geodesy/$4\pi$ normalization is $\bar P = P^{4\pi}/\sqrt2$.)
The recursion, vectorized identically to the Fortran, is

$$
\varepsilon(m,n) = \sqrt{\frac{(m+n)^2 - m^2}{4(m+n)^2 - 1}},
\qquad
\bar P_{m,0} = \sqrt{\tfrac{2m+1}{2m}}\,\cos\theta\,\bar P_{m-1,0},
$$
$$
\bar P_{m,1} = \frac{\mu\,\bar P_{m,0}}{\varepsilon(m,1)},
\qquad
\bar P_{m,n} = \frac{\mu\,\bar P_{m,n-1} - \varepsilon(m,n-1)\bar P_{m,n-2}}
                    {\varepsilon(m,n)}.
$$

## Storage layout and triangular truncation

Spectral coefficients live in a rectangular $(m, n)$ array with $m = 0\dots M$,
$n = 0 \dots M+1$, and **total wavenumber $l = m + n$**. For triangular
truncation $T_M$ (`num_fourier = M`, `num_spherical = M+1`):

- prognostic fields live on the triangle $l = m + n \le M$ (`prognostic_mask`);
- one extra diagonal $l = M + 1$ (`storage_mask`) is retained to support the
  meridional-derivative recurrences, which couple $n$ with $n\pm1$.

The Laplacian eigenvalues are $-l(l+1)/a^2$ (`laplacian_eigenvalues`). Standard
presets: `T42_GRID` (T42 at 64×128) and `T85_GRID` (T85 at 128×256), with Earth
radius $a = 6376$ km. `SpectralGrid` enforces `nlon > 2M` and
`nlat ≥ (2M+1)/2` so the transforms are exact (free of linear aliasing).

:::{admonition} Why the extra $l = M+1$ diagonal must be truncated after damping
:class: warning
Because `grid_to_spectral` retains the $l = M+1$ storage diagonal (needed for
$\partial/\partial\mu$), the prognostic tendencies are explicitly
triangular-truncated to $l \le M$ after damping. Left untruncated, those modes
are undamped grid-scale content that grows roughly exponentially at high
resolution and large timestep — the observed "T42, dt=600 s" instability.
Zeroing them restores per-step agreement with Isca to machine precision. See the
{doc}`../gallery/index` example *Triangular-truncation stability fix*.
:::

## Gaussian grid and quadrature

Gaussian latitudes $\mu_j = \sin\phi_j$ and quadrature weights are computed by
Newton iteration on the Legendre polynomials (the Numerical Recipes `gauleg`),
hemisphere-only with convergence $10^{-15}$
({py:func}`jsca.grid.gaussian`). The weights satisfy $\sum_j w_j = 2$, and the
global grid is assembled **south → north** (ascending $\mu$) — the jsca-wide
latitude convention.

## The transform pair

Longitude is handled by FFT, latitude by a dense Legendre matmul (`einsum`):

- **Analysis** (grid → spectral, `grid_to_spectral`): Fourier
  $F_m(\mu_j) = \tfrac{1}{\text{nlon}}\sum_i X\,e^{-i m \lambda_i}$ (`rfft/nlon`),
  then Gaussian quadrature

  $$
  X_l^m = \sum_j w_j\,F_m(\mu_j)\,\bar P_l^m(\mu_j).
  $$

  No extra normalization factor is needed because $\sum_j w_j = 2$ and
  $\int \bar P^2\,d\mu = 1$.

- **Synthesis** (spectral → grid, `spectral_to_grid`):
  $F_m(\mu_j) = \sum_l X_l^m\,\bar P_l^m(\mu_j)$, then `irfft(F·nlon)`.

The same module implements the spectral-space differential operators from
`spherical.F90`: the zonal derivative $\partial/\partial\lambda$ (coefficient
$m/a$), the meridional derivative $\partial/\partial\mu$ via the `dym`/`dyp`
recurrence coupling $n\pm1$, the curl/divergence $\alpha$ operators
(`compute_vor`, `compute_div`), and horizontal advection.
