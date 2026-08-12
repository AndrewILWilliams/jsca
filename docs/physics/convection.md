# Convection & condensation

`jsca` provides four selectable convection schemes plus large-scale
(stratiform) condensation. Convection is chosen through `convection_scheme`:
`SIMPLE_BETTS_MILLER` (the Frierson default), `FULL_BETTS_MILLER`, `DRY`
(Schneider–Walker), or `NONE`. Large-scale condensation runs alongside all of
them except `DRY`.

:::{admonition} Source
:class: note
{py:mod}`jsca.physics.qe_moist_convection`,
{py:mod}`jsca.physics.betts_miller`, {py:mod}`jsca.physics.dry_convection`,
{py:mod}`jsca.physics.lscale_cond`, {py:mod}`jsca.physics.sat_vapor_pres`.
:::

## Simplified Betts–Miller (`SIMPLE_BETTS_MILLER`)

The Frierson quasi-equilibrium scheme (`qe_moist_convection.F90`; Frierson 2007,
with virtual-temperature CAPE from O'Gorman & Schneider 2008). A two-stage
scheme: a CAPE/parcel diagnosis, then a Betts–Miller relaxation of $T$ and $q$
toward reference profiles.

### Stage 1 — parcel ascent

Per column, lift the lowest-level parcel:

1. If already saturated at the surface, the LCL is the surface; wring out
   moisture and warm the parcel by
   $T_p = T_0 + (r_0 - r_s)/\big[c_p/L_v + L_v r_s/(R_v T_0^2)\big]$.
2. Otherwise dry-adiabatic ascent conserving $\theta$; the LCL temperature is
   looked up from a table built once by Newton iteration. Below the LCL the
   parcel accumulates CIN.
3. At/above the LCL, **second-order (RK2 in $\ln p$) moist-adiabatic ascent**,
   with lapse coefficient

   $$
   \frac{dT}{d\ln p} = \frac{a}{1 + b},
   \qquad
   a = \kappa T + \frac{L_v}{c_p}r,
   \qquad
   b = \frac{L_v^2 r}{c_p R_v T^2}.
   $$

   CAPE and CIN accumulate from the **virtual**-temperature buoyancy
   $\pm R_d(T_{v,p} - T_{v,\text{env}})\ln\!\big(p_{\text{half}}(k{+}1)/p_{\text{half}}(k)\big)$,
   with $T_v = T\big(1 + q(R_v/R_d - 1)\big)$. Ascent stops at the level of zero
   buoyancy (LZB) or where $T_p < T_{\min}$.

### Stage 2 — Betts–Miller adjustment

Over the convective layer $[k_{\text{LZB}}, k_{\text{sfc}}]$, relax toward the
reference profiles $T_{\text{ref}} = T_p$ and a humidity from
$e_{\text{ref}} = \text{rhbm}\cdot p\,r_p/(r_p + \varepsilon)$:

$$
\Delta q = -(q_{\text{in}} - q_{\text{ref}})\frac{dt}{\tau_{\text{bm}}},
\qquad
\Delta T = -(T_{\text{in}} - T_{\text{ref}})\frac{dt}{\tau_{\text{bm}}}.
$$

The precipitation integrals $P_q = \tfrac1g\sum \Delta q\,\Delta p$ and
$P_t = \tfrac1g\sum \tfrac{c_p}{L_v}\Delta T\,\Delta p$ classify the column as
**deep** ($P_q, P_t > 0$; `convflag=2`), **shallow** ($P_t > 0, P_q \le 0$;
`convflag=1`, no precip), or inactive, with energy-conserving rescalings in the
deep case so column enthalpy is preserved.

**Key config:** `rhbm=0.7` (reference RH), `tau_bm=7200 s`, `Tmin=160 K`,
`Tmax=350 K`.

## Full Betts–Miller (`FULL_BETTS_MILLER`)

The classic scheme (`betts_miller.f90`; Betts & Miller 1986) — **distinct** from
the simplified one above. It has its own `capecalcnew` parcel ascent (a different
LCL formula, a hardcoded 127-entry LCL lookup table, and — unlike the qe scheme —
**no virtual-temperature effect** in CAPE/CIN), and its own reference-profile
relaxation with the energy-conserving timescale adjustment (`do_simp`). The
mixing ratio here uses $r = \varepsilon e_s/p$ throughout (dropping the $-e_s$ in
the denominator), and the moist-adiabatic ascent uses $T$ rather than $T_v$.

Only the default namelist path is ported (`do_simp=.true.`,
`do_shallower=.false.`, `do_changeqref=.false.`, `do_envsat=.false.`,
`do_taucape=.false.`, `buoyancy_kick=0`). **Key config:** `tau_bm=7200 s`,
`rhbm=0.8`, `do_simp=True`.

## Dry convection (`DRY`)

The Schneider–Walker dry adjustment (`dry_convection.f90`; Schneider & Walker
2006). It relaxes the temperature profile toward a prescribed lapse rate over a
timescale, touches **only temperature** (no moisture), and — when active — makes
the driver **skip large-scale condensation**. It returns a **rate** $dt_{tg}$
(unlike the Betts–Miller schemes, which return increments).

Lift a parcel from the lowest level at the prescribed lapse rate,

$$
T_p(k) = T_p(k{+}1) + \gamma\big(T_p(k{+}1)\,\zeta_k - T_p(k{+}1)\big),
\qquad
\zeta_k = \big(p_{\text{full}}(k)/p_{\text{full}}(k{+}1)\big)^{R_d/c_p},
$$

so $\gamma = 1$ follows the dry adiabat. Scan upward for the LCL, LZB and
CAPE/CIN; if $\text{CIN} > \text{CAPE}$, convection switches off. Where active,
shift the parcel profile by the mass-weighted mean $T_g - T_p$ so
column-integrated enthalpy is conserved, and set
$dt_{tg} = (T_p^{\text{adj}} - T_g)/\tau$.

**Key config:** `tau=21600 s`, `gamma=1.0`.

## Large-scale (stratiform) condensation

Removes supersaturation column by column (`lscale_cond.F90`, `do_simple=.true.`,
`do_evap=.true.`): where supersaturated, condense just enough vapour to return to
saturation (warming the layer by latent heat), let the condensate fall as rain,
and re-evaporate it into sub-saturated layers below. Returns **increments** and
column-integrated rain.

With $h_{lcp} = L_v/c_p$ and $q_{\text{sat}}, dq_{\text{sat}}/dT$ from
`sat_vapor_pres`, where $(q_{\text{in}} - q_{\text{sat}})q_{\text{sat}} > 0$:

$$
\Delta q = \frac{q_{\text{sat}} - q_{\text{in}}}{1 + h_{lcp}\,dq_{\text{sat}}}\;(<0),
\qquad
\Delta T = -h_{lcp}\,\Delta q\;(>0),
$$
$$
\text{precip} = \max\!\Big(-\sum_k \frac{p_{\text{half}}(k{+}1) - p_{\text{half}}(k)}{g}\,\Delta q(k),\;0\Big).
$$

**Re-evaporation** is a single top→surface pass carrying the falling rain mass:
condensing layers add to it, and sub-saturated layers with rain aloft moisten by
their saturation deficit (limited by available rain), cooling accordingly. **Key
config:** `hc=1.0`, `do_evap=True`.

## Saturation vapour pressure

The foundation for every moisture calculation (`sat_vapor_pres_k.F90`,
`do_simple`): constant-latent-heat Clausius–Clapeyron over liquid,

$$
e_s(T) = 610.78\,\exp\!\left[-\frac{L_v}{R_v}\left(\frac1T - \frac1{T_{\text{freeze}}}\right)\right],
\qquad
\frac{de_s}{dT} = \frac{L_v e_s}{R_v T^2}.
$$

The default saturation specific humidity (what Frierson uses) accounts for the
vapour's own partial pressure:

$$
q_s = \frac{\varepsilon e_s}{p - (1-\varepsilon)e_s},
\qquad
\frac{dq_s}{dT} = \frac{\varepsilon\,p\,(de_s/dT)}{\big(p - (1-\varepsilon)e_s\big)^2}.
$$

:::{admonition} Documented deviation
:class: note
Isca interpolates a precomputed $e_s$ table; `jsca` evaluates the closed form the
table is built from. The difference is only the table's interpolation error
($\sim 10^{-7}$ relative). All downstream scheme tendencies match Isca to that
level.
:::
