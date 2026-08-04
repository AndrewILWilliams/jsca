# Porting Isca to JAX — Scoping Document

**Prepared for:** Andrew · Project *PortingISCA*
**Date:** 2026-08-04
**Source analysed:** `Iscamaster 2.zip` (GitHub master snapshot of [ExeClim/Isca](https://github.com/ExeClim/Isca), GPL-3.0; no `.git` metadata in the zip, so the exact commit should be pinned in Phase 0)

**Decisions locked in (from our discussion):**

| Decision | Choice |
|---|---|
| Dynamical core | Faithful port of Isca's GFDL spectral core (Dinosaur/s2fft used as engineering references only) |
| Physics scope | Grey moist aquaplanet suite **and** RRTMG (MiMA-like) configurations; SOCRATES excluded behind an interface seam |
| Compute target | Single CPU node (multicore) first; GPU kept cheap to add later; multi-node MPI deferred |
| Staffing | Part-time / exploratory → plan structured as a feasibility spike with a go/no-go gate, then shippable phases |

---

## 0. Executive summary

The goal is a pure-Python, JAX-based reimplementation of Isca that (a) statistically reproduces the existing Fortran model against a pinned control simulation, (b) runs at Fortran-comparable speed on a multicore CPU node, and (c) can ultimately be coupled to VEROS to form an all-Python atmosphere–ocean model.

This is feasible, and there is direct precedent for every load-bearing claim. VEROS itself is the strongest one: it is a Fortran ocean model (pyOM2) rewritten in Python, and its documented benchmarks show the JAX backend running at speed comparable to the original Fortran on CPUs (NumPy, by contrast, is ~4× slower — JAX/XLA is what makes "pure Python but fast" true). On the atmosphere side, Google's Dinosaur dycore and the 2026 JCM model (Dinosaur + SPEEDY physics in JAX) prove the architecture. Nobody has done a *faithful* Isca port — JCM validated against SPEEDY only loosely (~1.8 °C surface-temperature RMS differences over 3-year runs). A port that reproduces Isca's climatology within internal variability, with a documented equivalence test, would be a genuinely novel and useful contribution — and it is also the prerequisite for trusting the coupled model later.

Three framing corrections to the request, stated plainly:

1. **"Rewrite the entire codebase" is the wrong target.** Of the ~343k lines of Fortran/C under `src/`, only ~30k lines are Isca-specific science for your chosen configurations (dycore + grey-suite physics), plus ~30k lines of executable RRTMG kernel code. About 116k lines are GFDL FMS infrastructure (MPI wrappers, diagnostics manager, time manager, I/O) whose correct fate is *replacement* by JAX/xarray/cftime, not porting; ~140k lines are RRTMG k-distribution coefficient tables that should become data files; and 63k lines of Fortran postprocessing collapse into a small xarray utility module. The true port surface is roughly **60k lines of Fortran → an estimated 15–25k lines of Python**.
2. **Bit-for-bit reproduction of the Fortran is impossible, and statistical reproduction is the correct standard** — your instinct (mean/std of a multi-year control) is the right one, and it needs a null distribution: two runs of the *Fortran* model that differ only by round-off produce different 5-year means. The test must therefore compare Fortran-vs-JAX differences against the Fortran model's own internal-variability envelope (Section 5). Tight numerical agreement is still enforced, but at the *module* level, where golden-input/output tests against instrumented Fortran can demand rtol ≈ 1e‑9–1e‑12.
3. **On CPU-first:** reasonable given your hardware, and VEROS shows JAX ≈ Fortran there. Two honest caveats: (i) per-core, mature `-O3` Fortran is a high bar, and at T42 the arrays are small enough that XLA's op-fusion and threading, not raw FLOPs, decide the outcome — this is exactly what the Phase-1 spike measures before you commit; (ii) the biggest wins JAX offers (GPU portability, `vmap` ensembles, differentiability) sit off the CPU path, so the design keeps them free even though we optimize CPU first.

At part-time intensity, expect roughly: dry spectral core validated against Held–Suarez in ~2 months of calendar time, the grey aquaplanet with a passing 5-year equivalence test in ~6–9 months, RRTMG in ~12–18 months, VEROS coupling after that. Every phase ends with something usable in research. If those horizons are unacceptable, the honest alternatives are more staffing or relaxing fidelity (adopt Dinosaur) — not optimism.

---

## 1. What is actually in the codebase (measured from your zip)

Line counts are `wc -l` over Fortran/C/include sources per directory:

| Component | Lines | What it is | Disposition |
|---|---:|---|---|
| `src/atmos_param/rrtm_radiation` | 169,788 | RRTMG_LW/SW radiation. **139,713 lines (82%) are `*_k_g.f90` coefficient DATA tables**; ~30k lines are executable kernels (`taumol`, `init`, solvers) | Data → NetCDF files; kernels → port (Phase 4) |
| `src/atmos_param` (rest) | ~31,500 | ~25 physics schemes (list below) | Port the subset your configs use (~13k); defer the rest |
| `src/atmos_spectral` | 13,348 | The spectral dycore (`spectral_dynamics`, `implicit`, `leapfrog`, `spectral_damping`, `fv_advection`, `water_borrowing`…) **plus** the solo driver: `idealized_moist_phys.F90` (1,471), `mixed_layer.F90` (824), `atmosphere.F90` (400) | Port faithfully (Phases 1–2) |
| `src/atmos_param/socrates` | 3,307 | Interface only — SOCRATES source is *not in the repo* (Met Office licence, separate download) | Excluded; keep radiation interface seam |
| `src/shared` | 115,669 | GFDL FMS: `mpp`, `diag_manager`, `time_manager`, `fft`, `sat_vapor_pres`, `astronomy`, `tridiagonal`, `topography`, `tracer_manager`, interp, … | **Replace** with Python stack; port only the small science kernels inside (sat-vapor-pressure tables, astronomy/insolation, Gaussian grid, tridiagonal solve, topography ingestion — ~3–5k lines) |
| `src/coupler` | 1,093 | `surface_flux` (Monin–Obukhov bulk fluxes) and flux-exchange scaffolding | `surface_flux` is science → port; scaffolding → replace |
| `src/atmos_column` | 1,474 | Single-column model driver | Port the *concept* — it becomes the physics test harness |
| `src/atmos_spectral_barotropic`, `_shallow` | 2,925 | Barotropic & shallow-water models | Optional; nice cheap validation targets for the transform layer |
| `src/atmos_solo`, `src/atmos_shared` | 4,805 | Drivers, tracer utilities, diagnostics glue | Mostly replaced by the Python driver |
| `postprocessing` | 63,050 | `mppnccombine`, pressure-level interpolation (Fortran) | **Replace**: xarray does tile-combining and vertical interpolation in ~500 lines |
| `src/extra/python/isca` + `exp` | ~11,600 (Py) | Experiment/namelist/diag-table management, 17 test cases (`held_suarez`, `frierson`, `MiMA`, `realistic_continents`, `bucket_hydrology`, `giant_planet`, `trip_test`…) | API concept carries over almost unchanged; compilation machinery disappears |

Physics inventory in `src/atmos_param` (key sizes): `two_stream_gray_rad` 809 · `qe_moist_convection` (simple Betts–Miller) 1,190 · `betts_miller` 895 · `lscale_cond` 313 · `cloud_simple` 2,829 · `monin_obukhov` 2,078 · `vert_diff` 1,089 · `vert_turb_driver` 804 · `diffusivity` 755 · `damping_driver` 642 · `dry_convection` 331 · `hs_forcing` 1,028 · `qflux` 97 · plus schemes your configs don't need (`ras` 5,207, `edt` 4,800, `entrain` 2,414, `my25_turb` 931, drag schemes, …) which are explicitly out of scope for v1.

Two useful discoveries in the tree: Isca ships a **`Dockerfile`**, which makes pinning the Fortran baseline easy, and a **`trip_test`** framework (bitwise comparison of two Fortran commits across a matrix of test configs) whose config matrix is exactly the coverage checklist the port should inherit.

---

## 2. Target architecture

Working name used below: **`jsca`** (rename at will). Python ≥3.11, JAX as the sole array backend, `float64` by default (`jax.config.update("jax_enable_x64", True)`) with a config switch to `float32` for production speed once validated.

### 2.1 Design principles

The deep structural change from the Fortran is not syntax — it is *state*. Fortran Isca keeps model state in mutable module variables, initialized by `_init` routines and mutated in place. JAX requires the opposite: **pure functions over an explicit, immutable state pytree**, with all configuration static and hashable so `jax.jit` can specialize on it. Everything else follows from four rules:

1. Every physical scheme is a pure function `(config, params, state_view) → tendencies/diagnostics`. No globals, no side effects, no I/O inside jitted code.
2. Model state is one registered dataclass (a pytree) threaded through the step function; leapfrog's two time levels are explicit leading array dimensions, not save-variables.
3. Static configuration (resolution, scheme choices, flags — the namelist content) lives in frozen dataclasses; runtime *numbers* (coefficients, k-tables, transform matrices) live in a `params` pytree of arrays, precomputed at init in float64.
4. Anything that is infrastructure in FMS (parallelism, diagnostics, calendars, restarts, I/O) is delegated to the ecosystem: XLA/`shard_map` for parallelism, xarray/zarr/NetCDF for I/O, `cftime` for calendars.

### 2.2 Package layout

```
jsca/
├── constants.py               # from FMS constants.F90, verbatim values
├── config/                    # frozen dataclasses mirroring namelists
│   └── compat.py              #   translator: Isca Namelist/Experiment → jsca configs
├── grid/
│   ├── gaussian.py            # Gaussian latitudes & weights (f64, from shared/…)
│   ├── transforms.py          # spherical harmonics: FFT (lon) + Legendre matmul (lat)
│   └── vertical.py            # sigma/hybrid coords, press_and_geopot port
├── dycore/                    # faithful port of src/atmos_spectral/model
│   ├── spectral_dynamics.py   # tendency assembly, exact Fortran call order
│   ├── implicit.py            # semi-implicit solve (implicit.F90 + matrix_invert.F90)
│   ├── leapfrog.py            # + Robert–Asselin filter
│   ├── damping.py             # ∇²ʰ spectral damping, sponge
│   ├── fv_advection.py        # vertical advection of tracers
│   └── water_borrowing.py
├── physics/
│   ├── api.py                 # Tendencies pytree, scheme protocol, driver ordering
│   ├── radiation/             # gray.py, byrne_ogorman.py, rrtmg/ (Phase 4), interface seam for SOCRATES-like externals
│   ├── convection/            # sbm.py (qe_moist_convection), betts_miller.py, dry.py
│   ├── condensation.py        # lscale_cond
│   ├── clouds_simple.py
│   ├── turbulence/            # vert_diff.py, diffusivity.py, vert_turb_driver.py, monin_obukhov.py
│   ├── surface/               # surface_flux.py, mixed_layer.py (slab + q-flux), bucket.py
│   └── forcing/               # hs_forcing.py (Held–Suarez), astronomy.py (insolation)
├── driver/                    # step(), lax.scan chunking, restarts, spin-up helpers
├── diagnostics/               # diag-table registry, on-device accumulation, xarray/zarr writer,
│                              #   plevel interpolation (replaces 63k lines of postprocessing)
├── experiments/               # held_suarez.py, frierson.py, mima.py, … mirroring exp/test_cases
├── coupling/                  # VEROS coupler (Phase 6): regrid weights, flux exchange
└── testing/                   # golden-fixture harness, statistical equivalence tools, benchmarks
```

### 2.3 State and the step function

```python
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ModelState:
    # spectral coefficients, complex128, shape (2, nlev, ncoeff) — 2 = leapfrog pair
    vor: Array; div: Array; temp: Array; lnps: Array   # lnps: (2, ncoeff)
    tracers: Array          # grid space, (2, ntracer, nlev, nlat, nlon) — q, …
    surface: SurfaceState   # t_surf, bucket depth, … (nlat, nlon)
    time: Array             # model seconds, int64

@functools.partial(jax.jit, static_argnames="cfg", donate_argnums=1)
def run_chunk(cfg: ModelConfig, state: ModelState, params: Params):
    def body(s, _):
        s = step(cfg, params, s)          # one Δt: physics → semi-implicit dynamics → filter
        return s, sample_diagnostics(cfg, s)
    state, samples = jax.lax.scan(body, state, xs=None, length=cfg.steps_per_chunk)
    return state, reduce_diagnostics(cfg, samples)   # e.g. daily means, still on device
```

`step` reproduces the Fortran call order exactly (from `atmosphere.F90` / `idealized_moist_phys.F90`): grid-space physics tendencies on the lagged time level, spectral transforms, semi-implicit adjustment, leapfrog with Robert–Asselin filter, spectral damping, tracer vertical advection and water borrowing. The outer Python loop runs one `run_chunk` per output interval (e.g. one day), writes diagnostics asynchronously, and checkpoints restarts as an exact-dtype pytree (zarr/npz).

### 2.4 The numerics being ported (so the fidelity contract is explicit)

Spectral representation — triangular truncation, Gaussian grid (T42L25 for the control):

$$X(\lambda_i,\mu_j)=\sum_{m=-M}^{M}\ \sum_{n=|m|}^{N}\ X_n^m\,\bar P_n^m(\mu_j)\,e^{\,im\lambda_i},$$

with analysis using Gaussian weights $w_j$. Longitude is an FFT (`jnp.fft.rfft`); latitude is a dense matmul against precomputed normalized associated Legendre functions $\bar P_n^m(\mu_j)$ — the formulation Dinosaur also uses, and the reason transforms are fast on both CPU (BLAS) and GPU. At T42, $\bar P$ is a few MB in float64; recurrences are evaluated once at init in float64.

Semi-implicit time stepping (`implicit.F90`): linearized gravity-wave terms $\mathcal{L}$ (divergence–temperature–surface-pressure coupling about a reference profile) are treated implicitly in the leapfrog step,

$$\frac{X^{t+\Delta t}-X^{t-\Delta t}}{2\Delta t}=\mathcal{N}\!\left(X^{t}\right)+\mathcal{L}\,\frac{(1+\epsilon)\,X^{t+\Delta t}+(1-\epsilon)\,X^{t-\Delta t}}{2},$$

which reduces, per zonal wavenumber and total wavenumber $n$, to small dense linear solves in the vertical (the `matrix_invert.F90` port — batched `jnp.linalg.solve` over precomputed factorizations). Time filtering is Robert–Asselin,

$$\overline{X}^{\,t}=X^{t}+\nu\left(\overline{X}^{\,t-\Delta t}-2X^{t}+X^{t+\Delta t}\right),$$

and horizontal dissipation is spectral hyperdiffusion (Isca default order $\nabla^8$), a per-coefficient multiplier

$$\left.\partial_t X_n^m\right|_{\rm diff}=-\nu_{2h}\left[\frac{n(n+1)}{a^2}\right]^{h} X_n^m .$$

All coefficients, reference profiles, truncation rules, and filter constants are read from the same namelist values as the Fortran, via the config translator.

Physics schemes are column-wise and vectorize trivially: written once over a `(..., nlev)` column and broadcast over `(nlat, nlon)` — no `vmap` gymnastics needed, though `vmap` gives ensembles for free later. The handful of iterative constructs (e.g. the Betts–Miller reference-profile adjustment, Monin–Obukhov stability iterations, saturation adjustment) become `lax.while_loop`/fixed-iteration loops with `where`-masking — the standard JAX patterns, and the part of physics porting that needs the most care in code review.

### 2.5 Configuration, diagnostics, experiments

Isca's front end is *already Python* — `Experiment`, `Namelist` (f90nml), `DiagTable` — and your muscle memory there should survive. `jsca.config.compat.from_namelist(...)` maps namelist groups onto the frozen config dataclasses and reports unmapped keys explicitly, so the existing `exp/test_cases/*.py` scripts port with edits measured in lines, not rewrites. The diag-table concept becomes a registry of `(field, reduction, interval)` requests compiled into the `sample/reduce` functions — accumulation happens on device inside the scan; writing happens outside jit via xarray → NetCDF/zarr with CF metadata, replacing `mppnccombine` and the Fortran plevel tools outright.

---

## 3. Keeping it fast (CPU node first)

### 3.1 Why "pure Python" can hold Fortran speed at all

At runtime Python only orchestrates: the entire model step is one XLA-compiled program (fused elementwise chains, BLAS matmuls, threaded via XLA's intra-op pool). The empirical anchor is VEROS's published benchmark page: their JAX backend is *"comparable to Fortran"* (pyOM2) on CPU, both single-node and under MPI scaling, while their NumPy backend — same Python, no XLA — runs ~4× slower. That gap **is** the design argument: the port must be JAX-idiomatic end-to-end (no NumPy islands, no Python in the hot loop), or it will be a 4×-slower model.

Honest caveats rather than promises: pyOM2/VEROS is finite-difference with large flat arrays, friendly to fusion; a T42 spectral model is *small* (a full 3-D field is ~1.6 MB in f64: 128×64×25), so per-op dispatch and cache behaviour dominate over FLOPs, and mature `-O3` Fortran on small arrays is a serious baseline. That is measurable, not arguable — hence the spike gate (§6). If single-core JAX lands within 0.5–1× of Fortran, the project is healthy (the wins that motivated JAX — ensembles, GPU, AD — come on top). Below ~0.3×, stop and diagnose before writing more physics.

### 3.2 The rules that keep it fast

**One jitted step, scanned.** `jax.jit` around the whole step; `lax.scan` over a day's worth of steps per dispatch. This amortizes Python/dispatch overhead to ~zero and gives XLA the whole-program view it needs to fuse. Never sync mid-chunk (no `.item()`, no printing, no per-step host callbacks).

**Precompute everything static.** Legendre matrices, Gaussian weights, semi-implicit vertical solve factorizations, damping multipliers, sat-vapor-pressure tables, (later) RRTMG k-tables — all built once at init, in float64, stored in `params`. The step function contains only array math.

**No recompilation.** Configs are frozen/hashable; shapes never change during a run; chunk length is fixed. Enable JAX's persistent compilation cache so the ~10s compile is paid once per config, not per job (JCM reports 10–15 s compiles for a comparable model — this is a non-issue if the cache is on).

**Memory discipline.** `donate_argnums` on the state so leapfrog buffers are reused; diagnostics reduced on device (daily means over a chunk, not per-step dumps); restarts and output written asynchronously outside jit.

**Precision as a switch.** Validate in float64. Once Tier-3 equivalence passes, offer float32 as a runtime mode (roughly 2× memory-bandwidth win on CPU, more on GPU); transforms/implicit solves stay prepared in f64 and cast down. NeuralGCM runs a comparable spectral core in f32 successfully, but *equivalence testing happens in f64* so precision never confounds a validation failure.

**Determinism.** On CPU, XLA is deterministic run-to-run for a fixed binary/thread configuration (disable fast-math-style flags in CI). This makes the golden-fixture tiers exactly reproducible, which is precisely what debugging a port needs — and a practical advantage of CPU-first for the validation years.

### 3.3 Parallelism, in the order you'll actually need it

1. **Single node, multicore (now).** XLA threads within ops (matmuls, fused kernels). At T42 the useful scaling ceiling per run is modest — expect saturation somewhere around 4–16 cores because the arrays are small; the spike measures the curve. Batching helps: the dycore's per-`m` solves and physics columns are expressed as big batched einsums, not loops, exactly so XLA has something to thread.
2. **Ensembles via `vmap` (free, and the best use of a fat node).** `vmap(step)` over N perturbed members turns one small model into one large, thread-friendly computation — often *better* node utilization than MPI-decomposing a single T42 run. The 5-member validation ensembles (§5) use this directly.
3. **GPU (cheap option later).** Zero code changes; expected to matter most for RRTMG's g-point-heavy columns and for ensembles/higher resolution. Worth a benchmark afternoon once Phase 2 lands, even if your day-to-day stays CPU.
4. **Multi-node CPU MPI (deferred, deliberately).** VEROS proves the `mpi4jax` route works and scales; but a spectral core needs distributed transposes for the transforms (what FMS `mpp` does), which is the most engineering for the least benefit at T42–T85 on one fat node. Decision explicitly revisited if you move to T170+, giant-planet grids, or coupled runs that outgrow the node. The layout keeps grid-space physics decomposition-friendly (`shard_map` over latitude bands) so the door stays open.

### 3.4 Benchmark harness (in the repo from day 1)

Phase 0 records the Fortran baseline **on your node** from the shipped Dockerfile: steps/second and years/day for `held_suarez` (T42L25), `frierson` (T42L25), and `MiMA` (T42L40), at 1, 4, 8, 16 cores. The Python side gets `pytest-benchmark` cases at the same granularity — transform round-trip, dry step, moist step, (later) RRTMG column batch — tracked in CI against those numbers, so a performance regression fails a test rather than being discovered in a paper deadline week. Targets: **≥0.5× Fortran node-for-node** to pass the spike; **≈1×** as the standing goal; anything above is upside.

---

## 4. Validation: what "reproduces the Fortran" will mean

### 4.1 Why not bitwise, precisely

Floating-point addition is not associative; XLA and gfortran make different fusion, FMA, vectorization, and reduction-order choices; the FFT implementations differ; libm functions (`exp`, `log`, `pow` in saturation formulae) differ in the last ulp. In a chaotic system any ulp difference grows to O(1) field differences within weeks of model time. Note that Isca's own `trip_test` only ever promises bitwise agreement *between two commits built with the same compiler on the same machine* — the Fortran model itself is not bitwise-stable across compilers or core counts. So the port's contract has two layers: **near-bitwise at the module level** (where chaos has no time to act), **statistical at the climate level** (where it does). This is the standard posture for climate-model verification — CESM's ensemble-consistency testing (Baker et al. 2015, `pyCECT`) is the formal reference for the climate layer, and we adopt a lightweight version of it.

### 4.2 Tier 0 — pin the baseline

Build the zip's source with its own `Dockerfile`; record compiler, flags, library versions, and (once you identify it) the upstream commit hash. Run and archive the control simulations below. Every later claim of equivalence refers to *this* frozen artifact, not "Isca" in the abstract. Deliverable: a `baseline/` directory with the container recipe, run configs, and archived output.

### 4.3 Tier 1 — golden module fixtures (the workhorse)

Instrument the Fortran with a small dump module (~200 lines: NetCDF writes of a routine's full inputs and outputs, callable at chosen timesteps) and harvest fixtures from a spun-up state for every ported routine: `qe_moist_convection`, `betts_miller`, `lscale_cond`, `two_stream_gray_rad`, `surface_flux`, `vert_diff`, `diffusivity`, `monin_obukhov`, `mixed_layer`, `damping`, the Gaussian grid, forward/inverse transforms, and the semi-implicit solve. (This is more robust than f2py-wrapping FMS-entangled routines.) A few hundred sampled columns per scheme, spanning regimes (tropics/extratropics, land/sea, day/night).

`pytest` then holds the port to tight tolerances in float64 — as a starting rule, rtol ≈ 1e‑12 for linear/algebraic kernels (transforms round-trip, damping multipliers, implicit solve residuals) and rtol ≈ 1e‑9…1e‑7 for transcendental-heavy chains (saturation adjustment, flux iteration), tightened or justified per module in review. This tier is what makes the rewrite *tractable*: chaos can't hide a porting bug in a single-call comparison, so by the time the full model runs, every part has already been proven equivalent in isolation.

### 4.4 Tier 2 — component and single-column tests

The `atmos_column` configuration is a gift: a single-column driver already exists in Isca, so full physics *chains* (radiation → convection → condensation → surface → diffusion over many steps) can be compared Fortran-vs-JAX without dynamical chaos, at tolerances between Tier 1 and Tier 3. Alongside it: Held–Suarez climatology for the dry core (1,200-day zonal-mean $\bar u(\phi,p)$, $\bar T(\phi,p)$ against the Fortran run and the published structure), optional barotropic/shallow-water checks of the transform layer, and continuous conservation audits (dry mass; water budget closure; spurious energy residuals) which catch a class of bugs statistics miss.

### 4.5 Tier 3 — the climate equivalence contract (your 5-year control)

**Control A (grey suite, the primary contract):** Frierson-style moist grey-radiation aquaplanet, T42, 25 levels — exactly the `frierson` test case in your tree — slab ocean with q-fluxes off (then a variant with them on), 2-year spin-up + 5-year analysis. **Control B (Phase 4):** MiMA-like RRTMG seasonal configuration, T42L40, same protocol.

Protocol, for each of Fortran and JAX: an ensemble of $N=5$ runs differing by an $O(10^{-6}\,\mathrm{K})$ temperature perturbation at $t=0$. Archive daily means of: $T$, $u$, $v$, $\omega$, $q$, RH on model levels; precipitation (convective/large-scale split), evaporation; surface fluxes (SH, LH, SW, LW), $T_s$; OLR; $p_s$. From these, the comparison set:

1. **Climatological means and standard deviations** of every archived field (annual and, for Control B, seasonal), zonal-mean cross-sections $\overline{X}(\phi, p)$, and maps.
2. **Circulation metrics:** jet latitude/strength, Hadley-cell edge and intensity ($\Psi_{500}$), eddy fluxes $\overline{u'v'}$, $\overline{v'T'}$, tropopause height.
3. **Variability:** daily-mean distributions (KS tests) of global means and selected point/zonal indices; precipitation PDF including wet-day tail; kinetic-energy spectra $E(n)$ (checks the damping/truncation port specifically).

**Pass criterion (proposed, to be frozen in Phase 0):** for each diagnostic, the JAX-minus-Fortran difference of ensemble means must lie within the Fortran ensemble's own internal-variability envelope — e.g. $|\Delta| \le 2\,\hat\sigma_{\rm ens}\sqrt{2/N}$ — with a false-discovery-rate correction at 5% across the diagnostic battery, *plus* absolute practical floors so a lucky wide envelope can't hide real drift (e.g. zonal-mean $T$ within 0.2 K, $u$ within 0.5 m s⁻¹, global precip within 2%). With $N=5$ this is a strict but attainable bar for a faithful port, and it is cheap: a T42 5-year run is hours on a node, so the full two-model, ten-run battery is an overnight-to-weekend job, rerunnable forever — which is what "a test case we can go back to" means.

**CI cadence:** Tier 1 on every commit (seconds); a 30-day Control-A smoke comparison weekly (catches gross regressions); the full Tier 3 battery at phase gates and releases.

---

## 5. Phased plan (calibrated to part-time / exploratory)

Assumed intensity: ~1–2 focused days/week; calendar estimates scale linearly with that. Each phase ends in something independently useful, so the project survives interruptions.

| Phase | Content | Gate to pass | Calendar (part-time) |
|---|---|---|---|
| **0. Baseline & harness** | Containerized Fortran build pinned; controls A(+B) run and archived; dump-module written; fixtures harvested; repo + CI + benchmark harness skeleton; equivalence thresholds frozen | Fortran baseline reproduces itself (trip-test style) and steps/s recorded on your node | 3–4 wk |
| **1. Feasibility spike: dry core** | Gaussian grid, transforms, semi-implicit leapfrog, damping, `hs_forcing`; Held–Suarez run | Tier-1 pass on transforms/implicit; HS climatology matches Fortran; **perf ≥0.5× Fortran node-for-node**. **GO/NO-GO decision here** | 6–8 wk |
| **2. Grey moist aquaplanet** | `qe_moist_convection`, `lscale_cond`, `two_stream_gray_rad`, `vert_diff`/`diffusivity`/`vert_turb_driver`, `monin_obukhov` + `surface_flux`, `mixed_layer` (+q-flux), water borrowing, spin-up tooling | **Control A Tier-3 equivalence passes** — the headline milestone | 3–5 mo |
| **3. Usability release v0.1** | Diagnostics manager, restarts, `compat.from_namelist`, docs; `betts_miller`, Byrne–O'Gorman radiation, `cloud_simple`, bucket land; port 3–4 `exp/test_cases` scripts verbatim | External user can run frierson-class experiments from a near-unchanged Isca script | 2–3 mo |
| **4. RRTMG** | k-tables → NetCDF (via a small Fortran dumper, not source parsing); port LW then SW kernels; per-band golden tests vs offline Fortran RRTMG columns (pyRTE‑RRTMGP as an independent cross-check oracle); seasonal astronomy; MiMA-like config | **Control B Tier-3 equivalence passes** | 4–6 mo |
| **5. Performance & platforms** | Multicore tuning against harness; f32 mode; GPU benchmark pass; (only if measurements demand) `mpi4jax`/`shard_map` multi-node | ≈1× Fortran node-for-node or a documented, understood gap; GPU numbers in hand | 1–2 mo |
| **6. VEROS coupling MVP** | §7 below | Coupled aquaplanet-with-continents run, closed budgets, stable multi-decade integration | 3–5 mo |

Sequenced honestly: **the grey-suite milestone (end of Phase 2) is ~6–9 months away at part-time; everything through RRTMG and coupling is a 1.5–2.5-year horizon.** If that timeline is unacceptable, the levers are staffing (a student/RSE roughly halves it — physics modules parallelize well across people once Phase 1 fixes the patterns) or scope (Dinosaur-core compromise), not compression of the validation work — that is the part that makes this project worth doing at all.

---

## 6. Coupling to VEROS

VEROS is the right partner for this: a Fortran ocean model (pyOM2) already rewritten in Python with a JAX backend, GPU support, and `mpi4jax`-based distribution — and, conveniently, GPL-3.0 like Isca, so the coupled system has a uniform licence. Its grid is a staggered lat-lon C-grid with ready-made global setups (e.g. 4°).

**Architecture: one Python process, sequential (lagged) coupling, both states on the same device.** No MCT/OASIS-style coupler infrastructure — the "coupler" is a Python module:

- **Atmosphere → ocean** each coupling interval (1–6 h, accumulated over atmospheric steps of ~600–1200 s): wind stress $(\tau_x,\tau_y)$, net surface heat flux components (SW, LW, sensible, latent), freshwater flux $(P-E)$.
- **Ocean → atmosphere:** SST on ocean points, replacing the slab `mixed_layer` temperature in `surface_flux`; the bucket land model continues to own land points. The clean seam already exists in Isca's design (the slab ocean is exactly the component being swapped out), and Phase 2's `mixed_layer.py` port should be written with that substitution in mind.
- **Regridding:** conservative remap weights between the T42 Gaussian grid (~2.8°) and the VEROS grid, computed once offline (xESMF/ESMF), applied in JAX as a precomputed sparse/dense matmul. At these resolutions the weights matrix is small; conservation is checked in CI (global flux integrals equal on both grids to round-off).
- **Time coupling:** standard asynchronous scheme — ocean steps (Δt ~ 1–3 h at 4°) consume time-accumulated fluxes; atmosphere sees SST held over the coupling window. Both models remain individually jitted; the exchange is a few small array ops between chunk calls, negligible cost.

**MVP experiment:** grey-radiation atmosphere (Phase 2 model) + VEROS global 4° setup, realistic continents on both sides, multi-decade integration with closed global heat and freshwater budgets as the acceptance test. Two open items to resolve early in Phase 6, flagged now: (i) VEROS has no dynamic sea-ice component as far as its docs show — the MVP needs a freezing-point SST clamp with an explicit flux treatment, and honest scope notes about high latitudes; (ii) grid-mismatch coastlines (T42 vs 4°) need a land-sea mask reconciliation policy. Neither is novel; both eat real time.

Worth saying: as far as I can find (searches in Aug 2026), no coupled atmosphere–ocean GCM in JAX exists — JCM is atmosphere-only, VEROS and FESOM2-JAX are ocean-only. Isca-JAX ⇄ VEROS would be the first, and a system paper in GMD/JAMES on its own.

---

## 7. Risks and mitigations

| Risk | Assessment | Mitigation |
|---|---|---|
| XLA:CPU underperforms Fortran on small T42 arrays | The one genuinely open technical question; VEROS evidence is favourable but finite-difference | Spike gate at Phase 1 with a hard number (≥0.5×); batching/einsum design; ensembles-per-node as the throughput story; GPU as the documented fallback |
| RRTMG port magnitude (~30k executable lines) | Largest single work item; mostly mechanical but relentless | 82% of it is data → NetCDF via a Fortran dumper; per-band golden tests; LW before SW; pyRTE‑RRTMGP as cross-check; grey-suite releases don't wait for it |
| Equivalence test ambiguity (small N, borderline fields) | Statistical tests can be gamed by wishful thresholds | Freeze metrics/thresholds in Phase 0 *before* any comparison; grow N only per pre-registered rule; publish the battery with the code |
| Part-time schedule decay | Certain, not hypothetical | Every phase ships a usable artifact; Tier-1 fixtures make context-switching cheap (tests tell you where you left off) |
| JAX API churn (`shard_map`, sharding APIs still evolving) | Moderate; core (`jit`/`scan`/`vmap`) is stable | Pin versions; isolate parallelism behind thin wrappers; CPU-first path uses only stable APIs |
| Iterative physics constructs under `jit` (BM adjustment, M-O iterations) | Known JAX friction: masking, `while_loop`, NaN debugging | Fixed-iteration ports where Fortran iteration counts are bounded; `checkify`/`jax_debug_nans` in CI; Tier-1 fixtures catch semantic drift immediately |
| Licensing | Low | Port of GPL Isca ⇒ new code is GPL-3.0 (VEROS matches); RRTMG carries AER's permissive notice inside — keep attribution headers; SOCRATES stays out |
| Fortran baseline bit-rot | Low but insidious | Container pinned in Phase 0; baseline outputs archived so the Fortran never needs rebuilding to re-validate |

---

## 8. Concrete first steps (next ~2 weeks of part-time effort)

1. Build the Fortran from your zip with the shipped `Dockerfile`; run `held_suarez` and `frierson` test cases; record steps/s at 1/4/8/16 cores on your node. Identify and pin the upstream commit.
2. Create the `jsca` repo skeleton (`pyproject`, `ruff`/`mypy`, `pytest`, CI, benchmark harness) — happy to generate this.
3. Write the Fortran dump module and harvest the first fixtures: Gaussian grid constants, one forward/inverse transform pair, one `implicit` solve.
4. Port `grid/gaussian.py` + `grid/transforms.py` and make the round-trip test pass at rtol 1e‑12 against those fixtures — the first real line of validated `jsca` code, and the start of Phase 1.

---

## 9. Appendix A — module-by-module port map (v1 scope)

| Fortran source | Lines | → Python target | Phase | Test approach |
|---|---:|---|---|---|
| `shared/constants`, `sat_vapor_pres`, `astronomy`, `tridiagonal` | ~3k of 115k (rest replaced) | `constants.py`, `physics/…`, `forcing/astronomy.py` | 1–2 | Tier 1 exact-value |
| `atmos_spectral/tools` (transforms, grid) | (in 13,348) | `grid/` | 1 | Round-trip vs fixtures, 1e‑12 |
| `atmos_spectral/model/{spectral_dynamics, implicit, leapfrog, spectral_damping, matrix_invert, press_and_geopot, fv_advection, water_borrowing}` | (in 13,348) | `dycore/` | 1–2 | Tier 1 per-routine + HS climatology |
| `hs_forcing` | 1,028 | `physics/forcing/hs_forcing.py` | 1 | Tier 1 + HS run |
| `atmos_spectral/driver/solo/{atmosphere, idealized_moist_phys}` | 1,871 | `driver/`, `physics/api.py` (call-order authority) | 2 | Tier 2 column chains |
| `two_stream_gray_rad` | 809 | `physics/radiation/gray.py` (+`byrne_ogorman.py`) | 2–3 | Tier 1 columns |
| `qe_moist_convection` / `betts_miller` / `dry_convection` | 1,190 / 895 / 331 | `physics/convection/` | 2–3 | Tier 1 columns, regime-stratified |
| `lscale_cond`, `cloud_simple` | 313, 2,829 | `condensation.py`, `clouds_simple.py` | 2–3 | Tier 1 columns |
| `vert_diff`, `diffusivity`, `vert_turb_driver`, `monin_obukhov` | 4,726 | `physics/turbulence/` | 2 | Tier 1 columns |
| `coupler/surface_flux`, `mixed_layer`, `qflux` | 1,093 + 824 + 97 | `physics/surface/` | 2 | Tier 1 + Tier 2; written for later SST substitution (VEROS) |
| `damping_driver` | 642 | `physics/damping_driver.py` | 2 | Tier 1 |
| Bucket hydrology (in `idealized_moist_phys`) + topography ingest | — | `physics/surface/bucket.py`, `grid/topography.py` | 3 | Tier 1 + realistic-continents smoke |
| `rrtm_radiation/rrtmg_{lw,sw}` kernels | ~30k exec (+139,713 data→NetCDF) | `physics/radiation/rrtmg/` | 4 | Per-band goldens; pyRTE‑RRTMGP cross-check |
| `atmos_column` driver | 1,474 | `testing/` SCM harness | 2 | — (it *is* the test) |
| Deferred: `ras`, `edt`, `entrain`, `my25_turb`, drag suite (`cg_drag`, `mg_drag`, `topo_drag`, …), `shallow_conv`, `strat_cloud`, SOCRATES interface | ~18k | — | post-v1 | — |

## Appendix B — references

- Isca: Vallis et al., *GMD* 11, 843–871 (2018), [doi:10.5194/gmd-11-843-2018](https://doi.org/10.5194/gmd-11-843-2018); [github.com/ExeClim/Isca](https://github.com/ExeClim/Isca)
- VEROS: [benchmarks](https://veros.readthedocs.io/en/v1.5.0/more/benchmarks.html) (JAX ≈ Fortran on CPU; NumPy ~4× slower); Häfner et al., *JAMES* (2021), [doi:10.1029/2021MS002717](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002717); [github.com/team-ocean/veros](https://github.com/team-ocean/veros) (GPL-3.0); advanced install (JAX/mpi4jax/GPU): [docs](https://veros.readthedocs.io/en/latest/introduction/advanced-installation.html)
- Dinosaur dycore: [github.com/neuralgcm/dinosaur](https://github.com/neuralgcm/dinosaur) (Apache-2.0, active — v1.3.6, June 2026); NeuralGCM: Kochkov et al., *Nature* 632 (2024), [doi:10.1038/s41586-024-07744-y](https://www.nature.com/articles/s41586-024-07744-y)
- JCM (closest prior art): Davenport et al., *GMD* 19, 6451 (2026), [article](https://gmd.copernicus.org/articles/19/6451/2026/) — Dinosaur + SPEEDY physics in JAX; T31 CPU ~0.05 SYPD, P100 ~2–3 SYPD; validated loosely vs SPEEDY (1.8 °C surface-T RMS)
- s2fft (JAX spherical harmonics, optional dependency/reference): [github.com/astro-informatics/s2fft](https://github.com/astro-informatics/s2fft); Price & McEwen, *J. Comp. Phys.* (2024), [article](https://www.sciencedirect.com/science/article/pii/S0021999124003589)
- mpi4jax: [github.com/mpi4jax/mpi4jax](https://github.com/mpi4jax/mpi4jax) (deferred multi-node path)
- pyRTE-RRTMGP (radiation cross-check oracle): [github.com/earth-system-radiation/pyRTE-RRTMGP](https://github.com/earth-system-radiation/pyRTE-RRTMGP)
- Ensemble consistency testing (Tier-3 methodology): Baker et al., *GMD* 8, 2829–2840 (2015), [doi:10.5194/gmd-8-2829-2015](https://doi.org/10.5194/gmd-8-2829-2015)
- FESOM2-JAX (evidence of the broader trend): [arXiv:2608.01546](https://arxiv.org/html/2608.01546)



