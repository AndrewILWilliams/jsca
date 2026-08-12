# Boundary layer & surface fluxes

The planetary boundary layer couples the surface to the free atmosphere. `jsca`
ports Isca's chain of four routines: Monin–Obukhov surface-layer similarity → the
bulk surface fluxes → the eddy-diffusivity $K$-profile → the implicit vertical
diffusion that mixes it all up the column.

:::{admonition} Source
:class: note
{py:mod}`jsca.physics.monin_obukhov`, {py:mod}`jsca.physics.surface_flux`,
{py:mod}`jsca.physics.diffusivity`, {py:mod}`jsca.physics.vert_diff`.
:::

## Monin–Obukhov surface-layer similarity

Relates surface turbulent exchange to near-surface stratification through
$\zeta = z/L$ ($L$ = Obukhov length) (`monin_obukhov_kernel.F90`). The scope is
`stable_option=1` (the Frierson default).

The differential similarity functions (with $b_{\text{stab}} = 1/\text{rich\_crit}$)
are, for unstable air ($\zeta < 0$), $\phi_m = (1 - 16\zeta)^{-1/4}$ and
$\phi_t = (1 - 16\zeta)^{-1/2}$; for stable air ($\zeta \ge 0$),
$\phi = 1 + \zeta(5 + b_{\text{stab}}\zeta)/(1 + \zeta)$. The integral functions
$f_m, f_t, f_q$ depend on $\zeta$, which itself depends on them — so `_solve_zeta`
**Newton-iterates** for $\zeta$ from the bulk Richardson number (up to 20 steps,
per-point convergence via `lax.fori_loop`).

`mo_drag` then returns the bulk drag coefficients and turbulent scales. With the
buoyancy contrast $\Delta b = g(\theta_0 - \theta)/\theta_0$ and
$\text{Ri} = -z\,\Delta b/(\text{speed}^2 + \text{small})$,

$$
u_* = \max\!\big(\text{vonkarm}/f_m,\;\sqrt{\text{drag\_min}}\big),
\quad
c_{d,m} = u_*^2,
\quad
c_{d,t} = u_* b_*,
\quad
c_{d,q} = u_* q_*.
$$

For $\text{Ri} \ge 0.95\,\text{rich\_crit}$ everything collapses to `drag_min`.
Companion routines `mo_diff` (height-dependent diffusivities) and `mo_profile`
(2 m / 10 m diagnostic interpolation ratios) round out the module. **Key config:**
`rich_crit=2.0`, `drag_min=1e-5`, `stable_option=1`.

## Bulk surface fluxes

Bulk aerodynamic fluxes plus the implicit derivatives the surface energy balance
needs (`surface_flux.F90`, Frierson `do_simple=.true.`, `old_dtaudv=.true.`).
With the surface saturation humidity, potential-temperature conversion, and
wind speed including a gustiness floor,

$$
\text{flux}_t = c_p\,(c_{d,t}w_{\text{atm}})\,\rho\,(T_{\text{surf}} - \theta_{\text{atm}})
\quad\text{(sensible heat)},
$$
$$
\text{flux}_q = (c_{d,q}w_{\text{atm}})\,\rho\,(q_{\text{surf},0} - q_{\text{atm}})
\quad\text{(evaporation)},
$$
$$
\text{flux}_u = (c_{d,m}w_{\text{atm}})\,\rho\,u_{\text{dif}},
\qquad
\text{flux}_r = \sigma T_{\text{surf}}^4.
$$

The implicit derivatives (`dhdt_surf`, `dhdt_atm`, `dedt_surf`, `dedq_atm`,
`drdt_surf` $= 4\sigma T_{\text{surf}}^3$, and the momentum-stress derivatives)
feed the mixed-layer solve.

**`use_virtual_temp`** (Frierson off; `column_test` on): with it on, the M–O drag
sees **virtual** potential temperatures $\theta_v = \theta(1 + d_{608}q)$ and the
density uses virtual $T$, so buoyancy accounts for moisture
($d_{608} = R_v/R_d - 1$).

**Bucket path** (`bucket=True`): a dry (empty) bucket sets
$q_{\text{surf},0} = q_{\text{atm}}$ — killing the humidity gradient, so there is
no evaporation; a wet surface stays saturated; and a **β-ramp** reduces land
evaporation once the bucket drops below 0.75 of capacity (see {doc}`surface`).

## PBL eddy diffusivity

Returns momentum/heat eddy diffusivities $k_m, k_t$ and the PBL depth $h$
(`diffusivity.F90`, the non-local $K$-profile scheme).

1. **Dry static energy over $c_p$:** `do_simple` → $\text{svcp} = T + (g/c_p)z$;
   Isca's default **`do_simple=.false.`** carries the virtual-temperature
   correction $\text{svcp} = T(1 + d_{608}q) + (g/c_p)z$.
2. **PBL depth:** the bulk Richardson number is walked upward until it crosses
   `rich_crit_pbl`, and $h$ is the interpolated crossing. On `do_simple=.false.`,
   **unstable** columns ($b_* > 0$) instead use a parcel-buoyancy top.
3. **$K$-profile:** below $h_{\text{inner}} = \text{frac\_inner}\cdot h$, use the
   M–O surface-layer $k$; between $h_{\text{inner}}$ and $h$ use the cubic

   $$
   K = K_{\text{ref}}\,\frac{z}{h_{\text{inner}}}
       \left(1 - \frac{z - h_{\text{inner}}}{h - h_{\text{inner}}}\right)^2,
   $$

   and above $h$ it is zero.

:::{admonition} `do_lcl_diffusivity_depth` — the column/SCM setting
:class: note
With `do_lcl_diffusivity_depth=.true.`, the PBL depth is instead the **convective
LCL height** (bypassing `svcp` and `pbl_depth` entirely). This was the dominant
single-column profile residual against Isca until it was ported — matching Isca's
`pbl_height` exactly. `DiffusivityParams.do_simple` now defaults to `False` to
match Isca's `diffusivity_nml`. **Key config:** `frac_inner=0.1`,
`rich_crit_pbl=1.0`, `parcel_buoy=2.0`.
:::

## Implicit vertical diffusion

Applies the diffusivities to mix $u, v, T, q$ up the column each step,
backward-Euler (`vert_diff.F90`, `gcm_vert_diff` down/up). The solve is **split
around the surface update**: `vert_diff_down` fully diffuses momentum (surface
stress as the implicit bottom BC, frictional dissipation fed into $dt_T$) and
forward-eliminates $T/q$; the mixed layer updates the surface RHS; `vert_diff_up`
back-substitutes $T/q$.

The implicit step is a tridiagonal system
$(1 - \delta t\,D)\,\xi^{n+1} = \xi^n + \text{explicit}$, with

$$
a = -\mu\,\nu(k{+}1)\,\delta t, \quad
c = -\mu\,\nu(k)\,\delta t, \quad
b = 1 - a - c,
$$

layer mass $\mu = g/\Delta p$, and diffusive conductance
$\nu = \rho_{\text{half}}K/\Delta z$, solved by Thomas elimination.
Energy-conserving **frictional heating**
$\text{diss\_heat} = -\tfrac1{c_p}\big[(u + \tfrac12\delta t\,du)\,du
+ (v + \tfrac12\delta t\,dv)\,dv\big]$ is added to $dt_T$. A `TriSurf` structure
carries the elimination coefficients and surface-coupling terms to the mixed
layer, which closes the $T/q$ surface boundary condition between the down and up
sweeps (`do_conserve_energy=.true.`, `do_virtual=.false.`).
