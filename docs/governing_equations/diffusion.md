# Horizontal diffusion & sponges

Spectral models need scale-selective damping at the truncation limit to remove
the enstrophy/energy that cascades to the grid scale and would otherwise
accumulate. `jsca` applies **hyperdiffusion** implicitly to the spectral
tendencies each step, plus optional top-of-model **sponge** layers.

:::{admonition} Source
:class: note
{py:mod}`jsca.dycore.spectral_damping` (ports `spectral_damping.F90`).
:::

## Implicit hyperdiffusion

Each step, the spectral tendencies are damped implicitly,

$$
\left.\partial_t X\right|_{\text{damped}} = \frac{\partial_t X - D\,X}{1 + D\,\Delta t},
$$

where $D$ (`damping_eff`) is a per-$(m,n)$ coefficient built once from the
positive Laplacian eigenvalues $\text{eigen} = l(l+1)/a^2$.

Three `damping_option` values are supported:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Option
  - Coefficient
* - **`resolution_dependent`**
  - $D = \text{coeff}\cdot\big(\text{eigen}/\text{eigen}(0, N-1)\big)^{\text{order}}$
    — normalized to the largest resolved total wavenumber.
* - **`resolution_independent`**
  - $D = \text{coeff}\cdot\text{eigen}^{\,\text{order}}$, i.e. the textbook
    hyperdiffusion multiplier
    $\left.\partial_t X_l^m\right|_{\text{diff}} = -\nu_{2h}\left[\tfrac{l(l+1)}{a^2}\right]^{h}X_l^m$.
* - **`exponential_cutoff`**
  - A Smith et al. (JFM 2002) filter on $\sqrt{\text{eigen}}$ above a cutoff
    wavenumber (zero below).
```

`damping_order` is the power $h$ on the *eigenvalue*, so the diffusion operator
is $\nabla^{2h}$.

:::{admonition} A note on "the default" damping order
:class: warning
`build_dynamics_params` sets `damping_order = 2` → $\nabla^4$ (`del^4`), with
`damping_coeff = 1.15740741e-4 s⁻¹ = (0.1 day)⁻¹`. Isca's *grey-physics*
configurations, however, run `damping_order = 4` → $\nabla^8$ (`del^8`), and the
Frierson climatology validation showed the higher order is what matches Isca —
`build_frierson` sets `damping_order = 4` (over-damping with order 2 gave cold
poles and weak high-latitude winds; see {doc}`../validation/index`). Both orders
are supported; the value in force is a per-run config choice, so always check the
model builder rather than assuming a single default.
:::

## Rayleigh drag and sponges

The generic field may additionally carry a linear (Rayleigh) drag
`+damping_coeff_r`, and vorticity/divergence may set independent `(coeff, order)`
pairs (`damping_vor`, `damping_div`).

Three sponge layers act **only on the top vertical level**, to absorb vertically
propagating waves near the model lid:

- an **eddy sponge** $= \text{eddy\_sponge\_coeff}\cdot\text{eigen}$ (for
  $m \ne 0$);
- **zonal-mean sponges** on the zonal-mean $u$ and $v$ ($m = 0$).

These are precombined into `sponge_vor` / `sponge_div` and applied with the same
implicit $(\partial_t X - s X)/(1 + s\,\Delta t)$ form on level 0 only.

## Sign convention

Isca's `get_eigen_laplacian` returns the *positive* $l(l+1)/a^2$, whereas
`jsca`'s `laplacian_eigenvalues` returns the negative; the sign is reconciled at
initialization (the negative eigenvalues are negated before being passed in), so
the damping coefficients match the Fortran exactly.
