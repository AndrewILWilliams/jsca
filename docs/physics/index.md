# Physics parameterizations

`jsca` ports Isca's **grey-radiation moist-aquaplanet** physics suite: the
column processes that supply the diabatic heating $J$, the moisture source, and
the momentum drag to the dynamical core. Each scheme is a faithful port of its
Isca ancestor, checked against golden Fortran fixtures, and each is documented
here with its governing equations, its scientific reference, and the Fortran
source it ports.

All column arrays store the **level axis last** (`k = 0` top … `k = K−1`
surface).

:::{admonition} Increments vs rates — a factor-of-Δt to keep straight
:class: important
The convection and condensation schemes (`qe_moist_convection`, `betts_miller`,
`lscale_cond`) return **increments** over the step — the driver divides by `dt`
to get a rate. Grey radiation and dry convection return **rates** directly. This
is Isca's convention, reproduced faithfully.
:::

## The `idealized_moist_phys` call order

For the Frierson configuration, the column-physics driver
({py:mod}`jsca.model.idealized_moist_phys`) composes the schemes in Isca's exact
order (F90 L819–1337). Grey radiation and vertical diffusion are each **split
into a down-pass and an up-pass around the mixed-layer surface update** — this is
how the implicit surface energy balance is closed.

1. **Convection** — relaxes $T, q$ toward a reference profile; returns heating,
   moistening, rain.
2. **Large-scale condensation** — removes supersaturation; latent heating +
   large-scale rain.
3. **Grey radiation (down)** — shortwave + downward longwave → surface fluxes.
4. **Surface fluxes** — bulk aerodynamic sensible/latent/momentum + implicit
   derivatives.
5. **Grey radiation (up)** — upward/net longwave; atmospheric radiative heating.
6. **Rayleigh sponge** — top-of-model drag (`damping_driver`).
7. **Boundary-layer diffusivities** — the $K$-profile.
8. **Implicit vertical diffusion (down)** — momentum fully; forward-eliminate
   $T, q$.
9. **Mixed layer** — slab-ocean surface energy balance; updates SST.
10. **Implicit vertical diffusion (up)** — back-substitute $T, q$ with the
    updated surface.

## The pages

```{toctree}
:maxdepth: 2

radiation
convection
boundary_layer
surface
held_suarez_forcing
```

## A note on `do_simple` and `sat_vapor_pres`

Frierson runs with `do_simple=True` throughout, selecting the simplified
constant-latent-heat Clausius–Clapeyron formulae. `jsca` ports the `do_simple`
branch; the full lookup-table paths are out of scope for this milestone. All of
the moisture-bearing schemes inherit a single documented $\sim 10^{-7}$
saturation-vapour deviation: Isca interpolates a precomputed $e_s$ table, while
`jsca` evaluates the closed form the table is built from. Every scheme's own
arithmetic is otherwise exact to machine precision against the Fortran.
