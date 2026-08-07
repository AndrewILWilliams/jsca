# Roadmap: the Frierson moist aquaplanet

> **Tracking issue: [#27](https://github.com/AndrewILWilliams/jsca/issues/27)** — the
> live checklist of module PRs and the definition of done (climatology parity vs
> Isca). This document is the detailed design; the issue is the progress board.

The dry Held-Suarez core is now validated against Isca. The next milestone is the
**Frierson (2006) idealized moist aquaplanet**: a moist GCM on a slab-ocean
aquaplanet with grey radiation, simplified Betts-Miller convection, large-scale
condensation, a bulk boundary layer, and surface fluxes. Target: statistically
reproduce the Isca `frierson_test_case` climatology (`exp/test_cases/frierson/`).

## What the config adds over Held-Suarez

Isca's `frierson_test_case.py` sets `idealized_moist_model=True`, which:

- makes **specific humidity `sphum` a prognostic tracer** advected by the
  spectral core (with `water_borrowing` for negativity — already ported);
- replaces the Newtonian/Rayleigh HS forcing with the **`idealized_moist_phys`**
  physics driver;
- uses `vert_coord_option='input'` (25 explicit Frierson hybrid levels),
  `dt_atmos=720`, `robert_coeff=0.03`.

## The `idealized_moist_phys` call order (per step, Frierson options)

Each column-physics module returns tendencies that accumulate into
`dt_tg`/`dt_qg`/`dt_ug`/`dt_vg`; the driver couples them to the surface through an
implicit vertical-diffusion solve.

1. **Convection** — `qe_moist_convection` (SIMPLE_BETTS_MILLER): relaxes T,q toward
   a moist adiabat / reference profile; returns convective heating, moistening,
   and rain.
2. **Large-scale condensation** — `lscale_cond`: removes supersaturation, latent
   heating + large-scale rain.
3. **Grey radiation (down)** — `two_stream_gray_rad_down`: Frierson two-stream SW
   + LW, p2 insolation; surface SW/LW down.
4. **Surface fluxes** — `surface_flux` (bulk aerodynamic + Monin-Obukhov): sensible
   heat, evaporation, momentum drag, and their implicit derivatives.
5. **Grey radiation (up)** — `two_stream_gray_rad_up`: upward/net LW; atmospheric
   radiative heating into `dt_tg`.
6. **Rayleigh sponge** — `damping_driver` (top-of-model drag).
7. **Boundary-layer K profiles** — `vert_turb_driver` / `diffusivity` (simple
   diffusivity).
8. **Implicit vertical diffusion (down)** — `gcm_vert_diff_down`: tridiagonal solve
   for u,v,T,q with the surface flux as an implicit bottom BC (`Tri_surf`).
9. **Mixed layer** — `mixed_layer`: slab-ocean surface energy balance, updates
   `t_surf`.
10. **Implicit vertical diffusion (up)** — `gcm_vert_diff_up`: completes the T,q
    diffusion using the updated surface.

## Port plan (dependency-ordered; each module gets Tier-1 Fortran fixtures)

**Phase A — column thermodynamics (self-contained, easy to fixture)**
1. `sat_vapor_pres` (do_simple) — es(T), qs(T,p). Foundation for 2, 3, 4. ✅ done (PR #19)
2. `lscale_cond` (do_simple, do_evap) — large-scale condensation. ✅ done (PR #20)
3. `qe_moist_convection` — simple Betts-Miller. Split in two (1190 lines):
   - 3a. **CAPE stage** — parcel ascent → CAPE, CIN, LCL/LZB levels. ✅ done (PR #21)
   - 3b. **Betts-Miller adjustment** — reference profiles, Pq/Pt, deep/shallow
     relaxation → `deltaT`/`deltaq`/`rain`/`convflag`. ✅ done ← this PR
4. `two_stream_gray_rad` — Frierson grey radiation + p2 insolation. ✅ done ← this PR
   **Phase A complete.**

**Phase B — surface & boundary layer** ← in progress
5. `monin_obukhov` + `surface_flux` (do_simple) — bulk fluxes.
   - 5a. `monin_obukhov` — surface-layer similarity (mo_drag + mo_profile). ✅ done (PR #24)
   - 5b. `surface_flux` (do_simple ocean) — bulk sensible/latent/momentum fluxes. ✅ done ← this PR
6. `diffusivity` / `vert_turb_driver` (do_diffusivity) — K profiles. ✅ done ← this PR
   (vert_turb_driver reduces to `diffusivity` + `gust=0`, `z_pbl=h` for Frierson.)
7. `vert_diff` (`gcm_vert_diff_down/up`) — implicit tridiagonal diffusion + surface coupling. ✅ done ← this PR
   (down/up split validated; the T/q surface coupling is closed by `mixed_layer`, item 8.)
8. `mixed_layer` — slab ocean. ✅ done
9. `damping_driver` — Rayleigh sponge. ✅ done

**Phase B complete.**

**Phase C — moist dynamics + assembly** ← in progress
10. Moist spectral core — `sphum` as a prognostic **grid** tracer.
    - 10a. Global **water-conservation correction** (`compute_corrections`,
      grid-tracer + MiMA pressure-limit path). ✅ done
    - 10b. **Grid tracer time-step** (`update_tracers` grid branch: physics
      tendency + `a_grid_horiz_advection` + `vert_advection` + Robert/RAW filter,
      incl. the L1248 dead-store quirk). ✅ done ← this PR (validated with
      `second_centered`; scheme-agnostic assembly).
    - 10c. **PPM vertical advection** (`finite_volume_parabolic`) in
      `vert_advection` — Frierson's production tracer vertical scheme. ✅ done ←
      this PR (parabolic reconstruction + Colella-Woodward limiter + Courant>1
      departure-point integral, validated to machine precision; the Courant>1
      walk is a loop-free `searchsorted`). `update_grid_tracer` slots it in via
      its `scheme` argument.

    **Item 10 (moist spectral core) complete.**
11. `idealized_moist_phys` assembly + the Frierson run + climatology validation
    vs Isca (`jsca.testing`).
    - 11a. **Column-physics driver** (`jsca.model.idealized_moist_phys`) —
      composes the ten validated column-physics modules in Isca's exact call
      order (F90 L819-1337): convection → condensation → grey-rad down → surface
      flux → grey-rad up → sponge → diffusivity → vert-diff down → mixed layer →
      vert-diff up. ✅ done ← this PR (stability/conservation smoke-tested; each
      component already golden-validated).
    - 11b. **Moist spectral stepping** (`jsca.model.frierson.FriersonModel`) —
      folds 11a into the spectral leapfrog loop with `sphum` as a grid tracer
      (`update_grid_tracer` + PPM + `water_correction`) and a slab-ocean `t_surf`.
      ✅ done ← this PR (the whole Frierson model runs end-to-end; stability
      smoke-tested). Caught & fixed a PPM `w=0` (0/0) robustness bug the unit
      fixture couldn't reach.
    - 11a fidelity gate. **Column driver validated against a real Isca step** —
      built + ran the pinned Isca Frierson at T42, instrumented the physics step,
      and matched jsca to machine precision (momentum ~1e-15; T/q at the
      `sat_vapor_pres` deviation ~1e-9). ✅ done. Caught & fixed two wiring bugs
      (real geopotential `z_half`; the `gust=1.0` initial gustiness).
    - 11c. **Frierson run + climatology vs Isca** — 🔶 **equilibrium benchmark
      done** (`docs/frierson_climatology.md`): T42 jsca `FriersonModel` runs vs a
      real Isca Frierson run, at two windows. At **200 days (days 100-200, near
      equilibrium)** the eddy field has matured and the dynamics now agree — u
      corr 0.97 (double jet 24.5 vs 38.1 m/s), precip corr 0.93 (ITCZ + storm
      tracks), T/q corr 0.99, t_surf corr 0.9993 — a large improvement on the
      60-day spin-up snapshot (u 0.73, precip 0.77). A roughly uniform ~7-10 K
      **cold offset** remains (t_surf −6.8 K, T −10.3 K), largest where radiative
      restoring is weakest; spin-up diagnostics show the poles/upper atmosphere
      still cooling at day 100 → likely incomplete high-latitude equilibration
      (per-step physics is machine-precision validated, excluding a physics bug).
      A longer 300-day run is under way to confirm. jsca runs ~1.4×/grid-point
      faster than single-core Isca on CPU (358 ms/step T42). **Full statistical
      parity (the #27 closer) still needs the cold-offset resolved + the
      `jsca.testing` ensemble verdict.**

Discovered Isca bug (documented in `vert_advection.py`): the PPM Courant>1 walk
for *downward* (`w<0`) interfaces exits on `kk==ks` while incrementing toward
`ke`, reading `dz(ke+1)` out of bounds. jsca clamps at `ke` (intended
behaviour); Frierson's sub-unity vertical Courant number never reaches that path.

Note: Frierson's `sphum` is a **grid** tracer (Isca `field_table`
`numerical_representation='grid'`), so the moist spectral core uses the grid
advection path (`a_grid_horiz_advection` + `vert_advection`), not the spectral
one. The water correction's spectral branch is therefore never reached and is
deliberately not ported.

Roughly one module per session. Fidelity is gated the same way as the dry core:
committed Fortran fixtures per routine, then the end-to-end climatology test.

## Note on `do_simple`

Frierson runs with `do_simple=True` throughout, which selects simplified
formulae (constant-latent-heat Clausius-Clapeyron for es, `qs = eps·es/p`, etc.).
We port the `do_simple` branch; the full lookup-table / exact paths are out of
scope for this milestone.
