# The spectral primitive equations

`jsca` integrates the hydrostatic primitive equations in the
**vorticity–divergence** formulation, the standard GFDL spectral-transform
discretization. This page gives the continuous equations and points at the code
that assembles each tendency.

:::{admonition} Source
:class: note
Tendency assembly: {py:func}`jsca.dycore.dynamics.compute_tendencies`
(ports `spectral_dynamics.F90` L845–910). Grid-space kernel:
`jsca.dycore.spectral_dynamics.four_in_one` (ports `four_in_one`, F90
L1038–1112). Spherical operators: {py:mod}`jsca.grid.transforms` (ports
`spherical.F90`).
:::

## Winds from vorticity and divergence

The horizontal winds are recovered from $\zeta$ and $\delta$ every step by
inverting the spectral vector operators. Writing $U = u\cos\phi$, $V = v\cos\phi$,
the relations

$$
\zeta = \frac{1}{a\cos\phi}\left(\frac{\partial V}{\partial\lambda}
        - \frac{\partial (U\cos\phi)}{\partial\mu}\right),
\qquad
\delta = \frac{1}{a\cos\phi}\left(\frac{\partial U}{\partial\lambda}
        + \frac{\partial (V\cos\phi)}{\partial\mu}\right)
$$

are inverted spectrally through the `uvm`, `uvc`, `uvp` recurrence coefficients

$$
\text{uvm} = \frac{-a\,\varepsilon_{m,n}}{l}, \quad
\text{uvc} = \frac{-a\,m}{l(l+1)}, \quad
\text{uvp}_n = \frac{-a\,\varepsilon_{m,n+1}}{l+1},
\qquad
\varepsilon_{m,n} = \sqrt{\frac{l^2 - m^2}{4l^2 - 1}},
$$

with $l = m + n$ (`compute_ucos_vcos`, `uv_grid_from_vor_div`).

## Momentum

In vector-invariant form, the momentum tendencies are

$$
\frac{\partial u}{\partial t} = (\zeta + f)\,v
  - \dot\sigma\frac{\partial u}{\partial\sigma}
  - R T\,\big(\nabla\ln p\big)_\lambda
  - \frac{1}{a\cos\phi}\frac{\partial}{\partial\lambda}
    \left(\Phi + \tfrac12(u^2 + v^2)\right),
$$

$$
\frac{\partial v}{\partial t} = -(\zeta + f)\,u
  - \dot\sigma\frac{\partial v}{\partial\sigma}
  - R T\,\big(\nabla\ln p\big)_\mu
  - \frac{1}{a}\frac{\partial}{\partial\mu}
    \left(\Phi + \tfrac12(u^2 + v^2)\right),
$$

with the Coriolis parameter $f = 2\Omega\sin\phi$. The pieces map to the code as:

- **Absolute-vorticity flux** — `abs_vort = ζ_grid + f`, then
  `dt_ug += abs_vort·v`, `dt_vg -= abs_vort·u`.
- **Pressure-gradient force** — `dt_ug -= R T · x2`, `dt_vg -= R T · x3` with
  $x2 = x1\,\partial_\lambda p_s$, $x3 = x1\,\partial_\mu p_s$ and the
  Simmons–Burridge coefficient

  $$
  x1_k = \frac{b_{k+1}\,\Delta\ln p^{(1)} + b_k\,\Delta\ln p^{(2)}}{\Delta p_k},
  \qquad
  \Delta\ln p^{(1)} = \ln p_{k+\frac12} - \ln p_k, \quad
  \Delta\ln p^{(2)} = \ln p_k - \ln p_{k-\frac12}.
  $$

- **Vertical advection** of $u, v$ via `vert_advection(..., ADVECTIVE_FORM)`.
- The vorticity and divergence tendencies are formed by taking the curl and
  divergence of $(\dot u, \dot v)$ in grid space
  (`vor_div_from_uv_grid`). The divergence tendency additionally carries the
  Laplacian of the geopotential-plus-kinetic-energy:

  $$
  \left.\frac{\partial\delta}{\partial t}\right|_{\text{p-grad}}
    = -\nabla^2\!\left(\Phi + \tfrac12(u^2 + v^2)\right),
  $$

  coded as `dt_divs -= laplacian(grid_to_spectral(Φ_full + ½(u²+v²)))`.

## Thermodynamics

$$
\frac{\partial T}{\partial t} = -\,\mathbf{u}\cdot\nabla T
  - \dot\sigma\frac{\partial T}{\partial\sigma}
  + \frac{\kappa T\,\omega}{p}
  + \frac{J}{c_p},
$$

where $\kappa = R/c_p$. The discrete adiabatic (omega-alpha) heating is formed in
`four_in_one`: `dt_tg -= κ T · x5` with

$$
x5_k = x4_k - u\,x2_k - v\,x3_k,
\qquad
x4_k = \frac{D^{<k}\,\Delta\ln p^{(3)}_k + D_k\,\Delta\ln p^{(1)}_k}{\Delta p_k},
$$

where

$$
D_k = \delta_k\,\Delta p_k + \Delta b_k\,(u\,\partial_\lambda p_s + v\,\partial_\mu p_s)
$$

is the mass divergence per layer (`dmean`), $D^{<k}$ its exclusive vertical
running sum (`dmean_excl`), and $\Delta\ln p^{(3)} = \ln p_{k+\frac12} -
\ln p_{k-\frac12}$. Horizontal temperature advection $-\mathbf{u}\cdot\nabla T$ is
added via `horizontal_advection`, vertical advection via `vert_advection`, and
the physics heating $J/c_p$ enters through `phys_dt_tg`. The full-level vertical
velocity is a by-product: $\omega_k = -x5_k\,p_k$.

## Moisture

Specific humidity $q$ (`sphum`) is a **grid** tracer. Its update
(`update_grid_tracer`, ports `update_tracers` grid branch, F90 L1223–1248) is

$$
q^\dagger = q^{t-\Delta t} + \Delta t\,\dot q_{\text{phys}},
$$

followed by A-grid horizontal advection (Lin–Rood finite volume,
{py:mod}`jsca.dycore.fv_advection`), vertical advection (the Frierson production
scheme is PPM, `FINITE_VOLUME_PARABOLIC`), and the RAW filter. Optional
hole-filling ({py:mod}`jsca.dycore.water_borrowing`) redistributes negative
humidity across the 5-point stencil while conserving total water; it is disabled
(`hole_filling='off'`) for Frierson.

## Surface-pressure tendency (continuity)

Vertically integrating the hybrid-coordinate continuity equation gives the
$\ln p_s$ tendency,

$$
\frac{\partial p_s}{\partial t} = -\sum_{k=1}^{K} D_k
\quad\Longrightarrow\quad
\frac{\partial \ln p_s}{\partial t} = \frac{1}{p_s}\frac{\partial p_s}{\partial t},
$$

coded as `dt_psg -= dmean_total` then `dt_ln_ps = grid_to_spectral(dt_psg / psg)`.
The interface vertical mass flux (the $\dot\eta$ analogue that drives vertical
advection) is

$$
\text{wg}_i = -\sum_{j<i} D_j + b_i\sum_j D_j,
\quad i = 1 \dots K-1,
\qquad \text{wg}_0 = \text{wg}_K = 0.
$$

## Hydrostatic geopotential

Geopotential is obtained by exact hydrostatic integration upward from the surface
({py:func}`jsca.dycore.press_and_geopot.compute_geopotential`):

$$
\Phi_{k+\frac12} = \Phi_{k+\frac32} + R\,T^v_k\,\big(\ln p_{k+\frac32} - \ln p_{k+\frac12}\big),
$$
$$
\Phi_k = \Phi_{k+\frac12} + R\,T^v_k\,\big(\ln p_{k+\frac12} - \ln p_k\big),
$$

seeded with $\Phi_{K+\frac12} = \Phi_s$ (surface geopotential). The virtual
temperature $T^v = T\,\big(1 + (R_v/R_d - 1)q\big)$ is used when
`use_virtual_temperature=True`, else $T^v = T$. When the model-top pressure is
zero (`pk[0] = 0`), $\Phi_{\frac12}$ is left at zero, matching the Fortran loop
bound.
