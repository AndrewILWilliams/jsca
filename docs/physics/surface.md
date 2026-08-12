# Surface: mixed layer & bucket hydrology

The lower boundary of the atmosphere is a **slab-ocean mixed layer** that sets
the surface temperature by an energy balance, optionally over a **land mask**
with a **Manabe soil-moisture bucket**.

:::{admonition} Source
:class: note
{py:mod}`jsca.physics.mixed_layer` (ports `mixed_layer.F90`),
{py:mod}`jsca.physics.bucket` (ports `idealized_moist_phys.F90` L1401–1428),
{py:mod}`jsca.model.land` (ports `land_generator_fn.py`). **Reference:** Manabe
(1969).
:::

## Slab-ocean mixed layer

Sits **between** the two vertical-diffusion sweeps and closes the implicit
surface energy balance, updating SST and the lowest-level $T/q$ increments.
Because the surface fluxes depend implicitly on the unknown surface and
lowest-level temperatures, the update is implicit. Using the vertical-diffusion
coupling terms ($dtmass$, $dflux$, stored $\delta t/\delta q$),

$$
\gamma_t = \frac{1}{1 - dtmass(dflux + \text{dhdt\_atm}/c_p)},
\qquad
\gamma_q = \frac{1}{1 - dtmass(dflux + \text{dedq\_atm})},
$$

the net downward surface flux and its $T_{\text{surf}}$-sensitivity give an
implicit heat-capacity update:

$$
C_{\text{eff}} = \text{depth}\cdot\rho c_p + \text{t\_surf\_dependence}\cdot dt,
\qquad
\Delta T_{\text{surf}} = \frac{-\text{corrected\_flux}\cdot dt}{C_{\text{eff}}}.
$$

The lowest-level $T$ and $q$ increments are then corrected with the new surface
temperature: $\delta t = f_{n,t} + e_{n,t}\Delta T_{\text{surf}}$,
$\delta q = f_{n,q} + e_{n,q}\Delta T_{\text{surf}}$. **Key config:**
`depth=2.5 m`, `albedo=0.31`, `evaporation=True` (Frierson slab).

### Land surface

With `land_option='input'` (the bucket configuration), two masked prefactors turn
ocean into land:

- `land_heat_capacity` scales the ocean $\text{depth}\cdot\rho c_p$ by
  `land_h_capacity_prefactor` over land (the bucket case uses 0.1, so land
  warms and cools faster than ocean);
- `land_albedo` scales the base albedo by `land_albedo_prefactor` over land
  (consumed by the radiation scheme).

Both are pure masked arithmetic; the aquaplanet path (`land=None`) is unchanged.

## The land mask

{py:func}`jsca.model.land.continents_land_mask` ports the `land_mode='continents'`
geometry from Isca's `land_generator_fn.py` — the Sauliere-2012 continent set
(N/S America, Eurasia, Africa, Australia, India, SE Asia; ≈18.8 % land). It is
validated bit-for-bit against that Fortran-adjacent Python on Isca's T42 grid.

:::{admonition} A note on the shipped bucket test case
:class: note
Isca's shipped `bucket_hydrology/input/land.nc` is actually a **square block** of
land (`land_mode='square'`), not continents. For the jsca-vs-Isca validation,
**both** models are driven from the same continents mask so the comparison stays
like-for-like.
:::

## Manabe bucket hydrology

A single-layer soil-moisture reservoir `bucket_depth` (metres of water) over
land, filled by precipitation and drained by evaporation, with excess above
capacity lost as runoff. It is marched with the **same grid-space leapfrog + RAW
filter** Isca uses for column tracers (its own coefficients
`robert_bucket=0.04`, `raw_bucket=0.53`):

$$
dt_{\text{bucket}} = \text{depth\_change\_cond} + \text{depth\_change\_conv}
    - \text{depth\_change\_lh},
$$
$$
\text{bucket\_depth} = \text{leapfrog+RAW}\big(\text{bucket\_depth}, dt_{\text{bucket}}, \delta t{=}1\big),
$$
$$
\text{bucket\_depth} \leftarrow \max(\cdot, 0),
\qquad
\text{bucket\_depth}^{\text{fut}} \leftarrow \min(\cdot, \text{max\_bucket\_depth\_land})\;\text{(land runoff)}.
$$

The three `depth_change_*` are already depths-per-step: the precipitation terms
are $\text{rain}/\rho_{\text{H}_2\text{O}}$ from the column physics, and the
evaporation term is $\text{flux}_q\,\Delta t/\rho_{\text{H}_2\text{O}}$ from the
surface-flux bucket path. Because $dt_{\text{bucket}}$ is the full increment, the
leapfrog runs with $\delta t = 1$. The reservoir is stored `(..., 2)` with the
leapfrog time level last, exactly like the humidity tracer. Evaporation off an
empty bucket is suppressed/β-ramped in the surface flux (see
{doc}`boundary_layer`). **Key config:** `max_bucket_depth_land=2 m`,
`init_bucket_depth_land=1 m`; the ocean reservoir is effectively infinite (always
wet).

The full land + bucket model is described in {doc}`../models/bucket` and validated
against Isca in {doc}`../validation/index`.
