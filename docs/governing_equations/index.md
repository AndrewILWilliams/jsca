# Governing equations

`jsca` solves the **hydrostatic primitive equations** in the vorticity–divergence
formulation on a hybrid σ/pressure vertical coordinate, discretized with the
**GFDL spectral transform method**: spherical-harmonic representation in the
horizontal, finite differences in the vertical, and a semi-implicit
leapfrog in time. This is a faithful port of the Isca / GFDL spectral core; every
equation on these pages is cross-referenced to its `jsca` module and the Fortran
source it ports.

Two conventions are used throughout and worth fixing in mind before reading:

- **Level index runs top → surface:** `k = 0` is the model top, `k = K` the
  surface — opposite to some textbook conventions.
- **Spectral storage is `(m, n)`** where `n` is an *offset* index; the **total
  (spherical) wavenumber is `l = m + n`**, and it is `l` that enters the
  Laplacian eigenvalue `−l(l+1)/a²`.

```{toctree}
:maxdepth: 2

primitive_equations
vertical_coordinate
spectral_representation
time_integration
diffusion
constants
```

## The prognostic variables

The spectral state advanced each step is:

```{list-table}
:header-rows: 1
:widths: 20 20 60

* - Symbol
  - Name
  - Representation
* - $\zeta$
  - relative vorticity
  - spectral $(m, n)$
* - $\delta$
  - horizontal divergence
  - spectral $(m, n)$
* - $T$
  - temperature
  - spectral $(m, n)$
* - $\ln p_s$
  - log surface pressure
  - spectral $(m, n)$
* - $q$
  - specific humidity
  - **grid** tracer $(\text{lat}, \text{lon}, k)$
```

Winds $u, v$ are diagnosed from $\zeta, \delta$ each step rather than stepped
directly. Humidity is carried on the grid (Isca's
`numerical_representation='grid'`), advected by finite-volume operators, not
spectrally — so the moist core uses the grid-tracer path throughout.

A defining property of the assembly: a **resting, isothermal, uniform-$p_s$
state produces identically zero tendencies**. A misplaced transpose or a sign
error breaks this exact steady state, so it is used as a unit invariant
(`tests/test_dynamics.py`).
