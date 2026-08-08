# Coupling jsca to Veros — a scoping note

**Question asked:** could we couple jsca (this JAX port of Isca) to
[Veros](https://veros.readthedocs.io) — the pure-Python/JAX ocean model — to
build an all-Python atmosphere–ocean GCM? What are the bottlenecks and issues,
and could we do it now?

This note answers from the *current* state of the port, not the state assumed
by the original scoping doc (`docs/scoping.md` §6), which placed coupling at
"Phase 6, after RRTMG". That placement is now stale: the moist atmosphere is
already validated, so the honest answer has moved a lot closer to "yes".

---

## Plain-language summary

- **The atmosphere side is ready enough to try.** jsca already runs a moist
  aquaplanet (the Frierson configuration) and reproduces the pinned Isca
  climatology to within its own internal variability at T21
  (`docs/frierson_climatology.md`). That is the milestone the old scoping doc
  said had to precede coupling — and it's done.
- **The place where the ocean plugs in is a single, clean function.** Today a
  2.5-m slab ocean (`mixed_layer_step`) sets the sea-surface temperature. Veros
  would take that job over. Everything the ocean needs (wind stress, heat-flux
  components, evaporation, rainfall) is *already computed* one step upstream and
  currently just handed to the slab. So the "coupler" is mostly a matter of
  redirecting numbers that already exist.
- **But "couple the validated model as-is" hits one hard wall: geography.** Our
  validated atmosphere is a *waterworld* — no continents, and permanent-equinox
  sunlight with no seasons. Veros's ready-made global-4° setup has *real*
  continents, bathymetry and (implicitly) seasons. You cannot bolt a waterworld
  atmosphere onto a real-continents ocean and get anything physical. So the
  realistic first coupled model needs work on **one** side to make the two
  agree on where the land is — and that, not the software plumbing, is the main
  cost.
- **Verdict:** a *toy* coupled run (aquaplanet atmosphere ⇄ an idealized
  aquaplanet-basin ocean) is a few focused weeks away and worth doing as a spike
  to prove the plumbing. A *scientifically meaningful* coupled run
  (Veros global-4° as-shipped) is a few months away, gated mostly by continents,
  seasonal insolation, sea ice and conservative regridding — none of it novel,
  all of it real time.

---

## 1. Where each side actually is

### jsca (this repo) — further along than `scoping.md` says

The `## State of the port` section of `CLAUDE.md` is out of date. What actually
exists and is fixture-validated:

- Full spectral **dycore** assembled and stepping: `dycore/spectral_dynamics.py`,
  `dynamics.py`, `implicit.py`, `leapfrog.py`, `spectral_damping.py`,
  `press_and_geopot.py`, `fv_advection.py`, `vert_advection.py`,
  `water_borrowing.py`, `global_integral.py`.
- Full **Frierson moist physics** stack: `physics/{qe_moist_convection,
  lscale_cond, two_stream_gray_rad, surface_flux, monin_obukhov, diffusivity,
  vert_diff, mixed_layer, damping_driver, sat_vapor_pres}.py`.
- An assembled, runnable **moist aquaplanet**: `model/frierson.py` +
  `model/idealized_moist_phys.py`, integrated with `lax.scan`.
- **Validated against pinned Isca** at T21: statistical parity (zero points fail
  the FDR-controlled ensemble-mean test on `u`, `T`, `sphum`, `t_surf`, precip),
  and ~1.6× single-core Fortran speed on the identical grid
  (`docs/frierson_climatology.md`).

So on the atmosphere side we are effectively at **end of the old Phase 2** — the
grey moist aquaplanet — which the original plan named as the prerequisite for
coupling. That prerequisite is met.

What is *not* there (and matters for coupling — see §5):

- **No continents / no realistic geography.** The model is
  `land_option='none'`: a saturated ocean everywhere, `z_surf = 0`, surface at
  rest. `idealized_moist_phys.py` hard-codes the aquaplanet
  (`land=.false.`, `u_surf=v_surf=0`).
- **No seasonal cycle.** Frierson grey radiation uses annual-mean,
  perpetual-equinox insolation. An ocean forced by that has no seasons to
  respond to.
- **No RRTMG, no realistic radiation** — irrelevant to the plumbing, relevant to
  whether the coupled climate means anything.

### Veros — the natural partner, with a known interface

Veros is pyOM2 (Fortran ocean) rewritten in Python with a JAX backend, GPL-3.0
(licence-compatible with our GPL-derived port). Its `global_4deg` setup is a
global ocean: roughly a **90 × 40 horizontal grid (4° lon × 4° lat), 15 vertical
levels**, tracer timestep on the order of **1–3 h**, with **real continents and
bathymetry**. (Confirm the exact `nx/ny/nz` and `dt_tracer` by reading the
shipped `set_parameters`/`set_grid` of the setup on the pinned Veros commit —
they are configuration, not physics.)

Crucially, Veros already exposes exactly the surface interface a coupler needs.
Its `set_forcing` routine, called every ocean step, sets:

- `surface_taux`, `surface_tauy` — zonal/meridional **wind stress**;
- `forc_temp_surface` — the **surface heat flux** into the top layer;
- `forc_salt_surface` — the **surface freshwater/salt flux**.

As shipped, `global_4deg` fills these from an ERA-40 monthly climatology, with
the heat flux applied as a **Haney-type restoring** (a climatological flux plus
`dQ/dT · (T_clim − SST)`) and sea-surface salinity **relaxed** to a climatology.
Coupling means *replacing* those climatological/restoring terms with jsca's
actual computed fluxes and **turning the restoring off** — otherwise the ocean is
still nudged to observations and the coupling is cosmetic.

Two Veros gaps to note up front: no dynamic **sea ice**, and the surface BC is
built around restoring, so removing it is part of the job.

---

## 2. The coupling seam is a single function

The cleanest fact in this whole analysis: the ocean's job in jsca today is done
by **one call**, and the state already carries the ocean temperature as a
first-class variable.

In `model/idealized_moist_phys.py`, the per-step physics runs 10 steps in Isca's
order. Step 9 is:

```python
# --- 9. slab-ocean surface energy balance (uses dt_real, not delta_t) ---
t_surf_new, _dts, tri = mixed_layer_step(
    params.mixed_layer, t_surf, sf.flux_t, sf.flux_q, sf.flux_r,
    net_surf_sw_down, surf_lw_down, sf.dhdt_surf, sf.dedt_surf, sf.drdt_surf,
    sf.dhdt_atm, sf.dedq_atm, tri, dt_real)
```

And `t_surf` is the 6th element of the model state threaded through
`model/frierson.py` (`state = (vors, divs, ts, ln_ps, qg, t_surf)`), rolled
forward each step like any prognostic.

To couple, you make two changes at this seam:

1. **Stop the slab from owning SST.** Replace step 9's `mixed_layer_step` with a
   "prescribed/external SST" path: `t_surf` for the coming atmospheric chunk is
   whatever Veros last handed back (held constant over the coupling window). Isca
   itself has this switch (`mixed_layer_bc` slab vs prescribed SST); we port the
   *concept*. `mixed_layer.py` and `surface_flux.py` were explicitly written with
   this substitution in mind (see `scoping.md` Appendix A).
2. **Harvest the fluxes for the ocean.** Every field Veros's `set_forcing` needs
   is already computed *before* step 9 and currently just consumed by the slab.
   Accumulate them over the coupling window and hand them out.

### What Veros needs vs. what jsca already computes

| Veros `set_forcing` field | jsca source (already computed each step) | in code |
|---|---|---|
| `surface_taux`, `surface_tauy` (wind stress) | surface momentum flux `sf.flux_u`, `sf.flux_v` | `surface_flux(...)` in `idealized_moist_phys` step 4 |
| net surface heat flux (`forc_temp_surface`) | `net_surf_sw_down` − `surf_lw_down`↓ + `flux_t` (sensible) + `flux_q·HLV` (latent) + `flux_r` (up-LW) | steps 3–5; these are exactly the terms `mixed_layer_step` already assembles into `corrected_flux` |
| freshwater flux `P − E` (`forc_salt_surface`) | `precip` − evaporation (`flux_q`) | `phys.precip` (output) and `sf.flux_q` |

So jsca **already produces the entire atmosphere→ocean flux vector**. The
coupler doesn't compute new physics; it redirects existing numbers. This is the
strongest single argument that coupling is close.

Direction ocean→atmosphere is even simpler: Veros's top-layer temperature,
regridded to the Gaussian grid, *is* the new `t_surf`.

---

## 3. Proposed architecture (one process, sequential/lagged)

Same shape the original scoping doc sketched (`scoping.md` §6), now concrete:

- **One Python process, both models on the same device.** No OASIS/MCT coupler;
  the "coupler" is a jsca module (`coupling/`).
- **Lagged (asynchronous) exchange at a coupling interval** of ~1–6 h,
  containing an integer number of atmospheric steps (`dt = 720 s` at Frierson
  T42) and ocean steps (~1–3 h):
  - jsca integrates one coupling window as a `lax.scan` chunk, **accumulating**
    the flux vector on-device (time-mean over the window — the scan already does
    exactly this pattern for climatology in `integrate_climatology`);
  - at the chunk boundary (host side, cheap), **regrid** the accumulated fluxes
    T42-Gaussian → Veros 90×40 and call one/several Veros steps;
  - **regrid** Veros SST → Gaussian and set it as `t_surf` for the next window.
- **Both models stay individually jitted.** The exchange is a handful of small
  array ops between chunk calls — negligible cost, provided we never sync
  *inside* either scan (the `bench/` and climatology code already respect this).

The refactor this needs on the jsca side:

1. A `coupling/` package: precomputed **regrid weight matrices** (dense/sparse
   matmul, built once offline with a conservative remap — ESMF/xESMF — and
   checked for global-integral conservation in CI, per `scoping.md` §6).
2. An **external-SST driver** variant of `model/frierson.py` that (a) takes
   `t_surf` as an input held over the window instead of updating the slab, and
   (b) returns the accumulated flux vector alongside the state. This is a small
   wrapper around `_step_full` — the seam is already isolated.
3. A **Veros runtime bridge**: instantiate a Veros setup whose `set_forcing`
   reads jsca's handed-in arrays instead of the ERA-40 climatology, with heat
   restoring disabled and freshwater from `P−E`.

---

## 4. Bottlenecks (performance / engineering)

1. **Two JAX runtimes in one process.** Veros wraps JAX behind its own backend
   abstraction (`runtime_settings`, NumPy/JAX switch, its own device/precision
   handling); jsca uses JAX directly with **x64 enabled globally**
   (`import jsca` sets `jax_enable_x64`). These must agree on device and dtype
   policy, and both jitted programs must coexist without thrashing the
   compilation cache. Likely a version-pinning + config-reconciliation chore, not
   a wall — but the first real integration friction. (`pyproject` currently pins
   only `jax>=0.4.35`; Veros pins its own range — reconcile.)
2. **Host handoff at each coupling interval.** By design the exchange sits at
   chunk boundaries, so the cost is small — *if* discipline holds. The danger is
   a `.item()`/host callback sneaking into the exchange and serializing the
   pipeline. Mitigation is the same rule the repo already follows: nothing
   host-side inside a scan.
3. **Timestep disparity** (atmos ~720 s, ocean ~1–3 h). Handled by the
   asynchronous scheme (ocean consumes a time-mean flux; atmosphere sees SST held
   over the window) — standard, not a blocker, but it sets the coupling interval
   and must divide evenly.
4. **Memory: both states resident.** At these resolutions both are small (a T42
   3-D field is ~1.6 MB f64; the 4° ocean is comparably tiny), so this is a
   non-issue on the CPU-first target and only matters if either side scales up.
5. **Regrid cost.** Negligible: a precomputed small matmul per exchange. The
   *construction* of conservative weights is offline and one-time.

None of these is the long pole. The long pole is scientific (§5).

---

## 5. Issues (scientific / correctness) — ranked

**1. Land–sea-mask mismatch — the dominant issue.** Our validated atmosphere is
a pure aquaplanet; Veros `global_4deg` has real continents. You cannot couple
them directly — the atmosphere has no land model, no orography, no coastline, and
Veros has no ocean under the continents. Two honest resolutions:

   - **(a) Idealized ocean basin for the spike.** Run Veros in an aquaplanet /
     idealized-basin configuration matching jsca's waterworld (or an
     all-ocean-except-a-simple-continent geometry). This proves the *plumbing*
     end-to-end with closed budgets and is the right **first** target. Cost:
     building/finding an aquaplanet Veros setup (the shipped one is realistic
     geography; an idealized basin is extra config).
   - **(b) Continents on the jsca side for the real target.** To use Veros
     `global_4deg` as-shipped, jsca needs land: a land-sea mask, the bucket land
     model + surface properties, topography ingestion, and a land–sea-mask
     reconciliation policy between the ~2.8° Gaussian grid and the 4° ocean grid
     (coastlines won't line up). This is real Phase-3-scale work
     (`scoping.md` Appendix A: bucket hydrology, `grid/topography.py`), none of it
     started.

**2. No seasonal cycle in the atmosphere.** Frierson uses annual-mean
perpetual-equinox insolation. An ocean's most interesting coupled behavior
(seasonal mixed-layer, sea-ice advance/retreat, seasonal SST) has nothing to
respond to. For a plumbing spike this is fine; for science you need seasonal
astronomy/insolation (`scoping.md` §4, Phase 4 astronomy) first.

**3. Removing Veros's restoring — or the coupling is fake.** `global_4deg`'s
heat BC is Haney restoring to ERA-40 and its salinity is relaxed to climatology.
If those stay on, the ocean is pinned to observations regardless of what the
atmosphere sends. Coupling *requires* replacing `forc_temp_surface` with jsca's
net flux and disabling the restoring; freshwater `P−E` replaces the salt
relaxation. This changes the ocean's stability properties (a freely-evolving SST
can drift), which is exactly why budget closure (below) is the acceptance test.

**4. No sea ice, and grey radiation at high latitudes.** Veros has no dynamic
sea ice; the shipped restoring hides that. Under a free flux from a grey
atmosphere, high-latitude SST can drift below freezing. Needs a **freezing-point
SST clamp with explicit flux accounting** and honest scope notes about polar
behavior (flagged already in `scoping.md` §6).

**5. Conservation across the regrid.** Energy and freshwater must be conserved
in the T42↔4° remap or the coupled system drifts spuriously. Use **conservative**
remap weights and assert global-integral equality on both grids in CI
(round-off). This is the primary correctness gate of the whole exercise.

**6. Stability of free SST feedback.** Replacing a 2.5 m slab (fast, stabilizing)
with a full-depth ocean (slow, with its own variability and no restoring) changes
the coupled feedback. Expect a multi-decade drift/spin-up phase; the acceptance
test is **closed global heat and freshwater budgets over a multi-decade
integration** (as `scoping.md` §6 states), not short-run agreement.

---

## 6. Could we do it *now*?

Split the question, because the two answers are very different:

**A "plumbing spike" — yes, weeks, and worth doing.** Aquaplanet jsca ⇄ an
idealized-basin Veros, exchanging real fluxes, with closed budgets as the test.
This proves the hardest *unknowns* cheaply: two JAX runtimes co-resident, the
external-SST seam, conservative regridding, and asynchronous exchange. It needs:

  1. `coupling/` with conservative regrid weights + a CI conservation check;
  2. an external-SST driver wrapping `_step_full` (the seam is already isolated
     at step 9 — small);
  3. a Veros bridge with `set_forcing` reading jsca arrays and restoring off;
  4. an idealized-basin Veros setup;
  5. a short coupled run showing closed global heat/freshwater budgets.

Nothing here needs new atmospheric physics — the validated model suffices.

**A scientifically meaningful coupled GCM — no, not now; a few months.** Using
Veros `global_4deg` as-shipped is gated by, in rough order: **continents + land
model + topography on the jsca side** (issue 1b), **seasonal insolation**
(issue 2), **sea-ice/freezing policy** (issue 4), then the same plumbing as the
spike. These are real, mostly-non-novel tasks; they are exactly the Phase-3/4
items the original scoping doc listed, now on the critical path for a *useful*
coupled model rather than for RRTMG.

### Recommended sequence

1. **Do the plumbing spike first** (aquaplanet ⇄ idealized basin). It de-risks
   every software unknown and produces a demonstrable all-JAX coupled ocean–
   atmosphere run — itself a novel artifact (no coupled atmosphere–ocean GCM in
   JAX exists as of the scoping survey; `scoping.md` §6).
2. **In parallel, cost the geography work** — the land/continents effort is the
   real budget item and can be scoped independently of the coupler.
3. **Only then target `global_4deg` as-shipped**, with the budget-closure
   acceptance test.

The one-line answer to "could we do it now": **the plumbing is ready and the
atmosphere is validated, so a demonstrator coupled run is close; a realistic
coupled climate is gated by continents and seasons, not by the coupling
machinery.**

---

## Open questions to pin before starting

- Exact Veros commit to pin (mirror the Isca-pinning discipline), and its
  `global_4deg` `nx/ny/nz/dt_tracer` — read from source, not docs.
- JAX version reconciliation between jsca (x64 global) and the pinned Veros.
- Idealized-basin Veros setup: does a suitable one exist, or must we build it?
- Conservative-remap tooling choice (xESMF/ESMF offline) and how weights are
  stored (npz/zarr) and loaded into the jitted path.
- Coupling interval and whether the atmosphere sub-steps the ocean or vice-versa
  at the chosen resolutions.

## References

- Current jsca validation: `docs/frierson_climatology.md` (T21 statistical
  parity with pinned Isca; ~1.6× Fortran speed).
- Coupling seam: `src/jsca/model/idealized_moist_phys.py` (step 9,
  `mixed_layer_step`); `src/jsca/model/frierson.py` (`t_surf` in model state).
- Original architecture sketch: `docs/scoping.md` §6 and Appendix A.
- Veros: <https://veros.readthedocs.io>, `global_4deg` setup;
  Häfner et al., *JAMES* (2021), doi:10.1029/2021MS002717;
  <https://github.com/team-ocean/veros> (GPL-3.0).
