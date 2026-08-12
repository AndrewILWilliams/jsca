# Held–Suarez (dry benchmark)

The dry Held–Suarez (1994) benchmark integrates the spectral dynamical core
forced only by Newtonian thermal relaxation and Rayleigh drag — no moisture, no
radiation, no boundary-layer physics. It is the standard dry-dynamical-core test
and `jsca`'s first end-to-end fidelity gate.

:::{admonition} Source
:class: note
{py:mod}`jsca.model.held_suarez`, forced by {py:mod}`jsca.physics.hs_forcing`
(see {doc}`../physics/held_suarez_forcing`).
:::

## What it steps

The prognostic state is the dry spectral core: vorticity $\zeta$, divergence
$\delta$, temperature $T$, and log surface pressure $\ln p_s$ (with leapfrog time
levels). Each step:

1. compute the adiabatic dynamical tendencies (see
   {doc}`../governing_equations/primitive_equations`);
2. add the Held–Suarez forcing — Newtonian $-k_T(T - T_{\text{eq}})$ and Rayleigh
   $-k_v(u, v)$ with frictional heating;
3. apply the semi-implicit correction, spectral damping, and the RAW leapfrog.

## Usage

```python
from jsca.model import held_suarez as HS

model  = HS.build_held_suarez(num_fourier=42)   # T42, 25 uneven-sigma levels
state0 = HS.initial_state(model)                # resting isothermal (264 K)

# Integrate, optionally sampling zonal-mean fields every N steps.
state, samples = HS.integrate(model, state0, n_steps=..., sample_every=...)
```

The native benchmark configuration is **T42, 25 uneven-sigma levels,
`damping_order=4`, dt = 600 s** — the exact config Isca ships. Running there is
what the triangular-truncation stability fix (see
{doc}`../governing_equations/spectral_representation`) made possible.

## Validation

The dry HS benchmark statistically reproduces the pinned Isca climatology. Both
models integrate from a resting isothermal state, discard 120 days of spin-up,
and form eight 30-day monthly means as an ensemble; the Tier-3
`ensemble_mean_test` asks whether jsca's ensemble mean sits within Isca's own
month-to-month spread at every (level, latitude) point.

```{list-table}
:header-rows: 1
:widths: 20 20 20 20 20

* - Field
  - bias
  - RMS
  - max |diff|
  - points failing
* - zonal wind $u$
  - −0.09 m/s
  - 0.69 m/s
  - 3.1 m/s
  - **0.0 %**
* - temperature $T$
  - −0.02 K
  - 0.31 K
  - 1.4 K
  - **0.0 %**
```

The eddy-driven jet strength is **Isca 33.6 m/s, jsca 33.7 m/s** (±40°, 250 hPa),
and no point differs from Isca beyond its internal variability. See
{doc}`../validation/index` and the gallery example
{doc}`../gallery/index`.
