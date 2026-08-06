# Roadmap: the Frierson moist aquaplanet

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
6. `diffusivity` / `vert_turb_driver` (do_diffusivity) — K profiles. ← next
7. `vert_diff` (`gcm_vert_diff_down/up`) — implicit tridiagonal diffusion + surface coupling.
8. `mixed_layer` — slab ocean.
9. `damping_driver` — Rayleigh sponge.

**Phase C — moist dynamics + assembly**
10. Moist spectral core — `sphum` as a prognostic tracer (spectral advection +
    `water_borrowing`, both partly ported); `four_in_one` tracer handling.
11. `idealized_moist_phys` assembly + the Frierson run + climatology validation
    vs Isca (`jsca.testing`).

Roughly one module per session. Fidelity is gated the same way as the dry core:
committed Fortran fixtures per routine, then the end-to-end climatology test.

## Note on `do_simple`

Frierson runs with `do_simple=True` throughout, which selects simplified
formulae (constant-latent-heat Clausius-Clapeyron for es, `qs = eps·es/p`, etc.).
We port the `do_simple` branch; the full lookup-table / exact paths are out of
scope for this milestone.
