# Held–Suarez forcing

The Held–Suarez (1994) benchmark replaces the full physics with an analytic
forcing: **Newtonian relaxation** of temperature toward a radiative-equilibrium
profile and **Rayleigh damping** of near-surface winds. It is the standard
dry-dynamical-core test, and `jsca`'s first end-to-end fidelity gate.

:::{admonition} Source
:class: note
{py:mod}`jsca.physics.hs_forcing` (ports `hs_forcing.F90`, default
`equilibrium_t_option='Held_Suarez'`, `do_conserve_energy=.true.`). **Reference:**
Held & Suarez (1994).
:::

## Newtonian thermal damping

The equilibrium temperature with a stratospheric floor is

$$
T_{\text{eq}} = \max\!\left[\Big(T^\ast - \delta_v\cos^2\phi\,\ln(p/p_{00})\Big)
    \Big(\frac{p}{p_{00}}\Big)^\kappa,\; T_{\text{strat}}\right],
$$
$$
T^\ast = T_0 - \delta_h\sin^2\phi - \varepsilon\sin\phi,
\qquad
T_{\text{strat}} = T_{\text{strat},0} - \varepsilon\sin\phi,
$$

and the relaxation rate has a boundary-layer enhancement (for
$\sigma_b < \sigma \le 1$, with $\sigma = p/p_s$):

$$
k_T = k_a + (k_s - k_a)\cos^4\phi\;\frac{\sigma - \sigma_b}{1 - \sigma_b}
\quad\text{(in the PBL)},
\qquad
k_T = k_a \quad\text{above},
$$
$$
\left(\frac{\partial T}{\partial t}\right) = -k_T\,(T - T_{\text{eq}}).
$$

## Rayleigh damping

Near-surface winds are damped in the boundary layer only:

$$
k_v = k_f\,\frac{\sigma - \sigma_b}{1 - \sigma_b}
\quad (\sigma_b < \sigma \le 1),
\qquad
\frac{\partial u}{\partial t} = -k_v u,
\qquad
\frac{\partial v}{\partial t} = -k_v v.
$$

## Frictional heating

With `do_conserve_energy` (using the previous-step winds $u_m, v_m$), the kinetic
energy removed by Rayleigh drag is returned as heat:

$$
\left(\frac{\partial T}{\partial t}\right)_{\text{fric}}
  = -\frac{(u_m + \tfrac12\dot u\,dt)\dot u + (v_m + \tfrac12\dot v\,dt)\dot v}{c_p}.
$$

## Key configuration

```{list-table} `HsForcingParams` (`hs_forcing_nml` defaults)
:header-rows: 1
:widths: 22 18 60

* - Field
  - Default
  - Meaning
* - `t_zero` $T_0$
  - 315 K
  - Maximum equilibrium temperature.
* - `t_strat`
  - 200 K
  - Stratospheric floor.
* - `delh` $\delta_h$
  - 60 K
  - Equator–pole equilibrium $\Delta T$.
* - `delv` $\delta_v$
  - 10 K
  - Static-stability parameter.
* - `eps` $\varepsilon$
  - 0
  - Hemispheric asymmetry.
* - `sigma_b`
  - 0.7
  - PBL top (in σ).
* - `ka`, `ks`, `kf`
  - −40, −4, −1 d
  - Damping times in days (negative), converted to rates.
```

Only the default HS configuration is ported — the `from_file`, `exoplanet`, and
`top_down` equilibrium-temperature variants, specified-wind relaxation, local
heating, and tracer forcing are out of scope. See {doc}`../models/held_suarez`
for the model assembly and {doc}`../validation/index` for the jsca-vs-Isca
climatology.
