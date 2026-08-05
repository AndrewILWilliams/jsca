# Isca Held-Suarez reference climatology

`hs_isca_reference.npz` is the pinned **Fortran** climatology that jsca's
Held-Suarez benchmark must statistically reproduce. It was produced by the
actual Isca dynamical core + Newtonian-cooling forcing — not by jsca — so it is
a legitimate validation target under the "fixtures come only from Fortran" rule
(CLAUDE.md). The dry HS94 benchmark is the first end-to-end fidelity gate for
the whole port.

## How it was generated

1. **Model:** Isca `held_suarez.x`, built from the pinned snapshot
   (`ExeClim/Isca` commit
   `a290bc376d84d0ee83adbb80eb374b9f629c3534`; see `baseline/PINNED.md` for the
   toolchain and the required gfortran-13 flags).
2. **Config:** the shipped `exp/test_cases/held_suarez/held_suarez_test_case.py`
   verbatim — **T42, 25 levels**, `dt_atmos = 600 s`, `damping_order = 4`,
   `vert_coord_option = 'uneven_sigma'` (`scale_heights=6`, `exponent=7.5`,
   `surf_res=0.5`), `valid_range_t=[100,800]`, HS forcing defaults
   (`t_zero=315`, `delh=60`, `delv=10`, `sigma_b=0.7`, `ka=-40`, `ks=-4`,
   `kf=-1`). Reproduce with `bench/run_isca_held_suarez.py`.
3. **Integration:** 12 × 30-day months from a resting isothermal start. The
   first **4 months (120 days)** are discarded as spin-up; the reference is the
   **time + zonal mean of months 5–12 (240 days)**. Built with
   `baseline/make_hs_reference.py`.

The only deviations from the shipped recipe are sandbox-portability knobs
(`num_cores=4` instead of 16, `mpirun --allow-run-as-root --oversubscribe`).
Core count changes the domain decomposition and hence bit-level roundoff, but
not the statistical climatology that jsca is validated against.

## Contents

| key | shape | meaning |
|---|---|---|
| `lat` | (64,) | Gaussian latitudes, °N (south→north) |
| `pfull` | (25,) | full-level pressure, hPa |
| `u_zm`, `t_zm`, `v_zm` | (25, 64) | time+zonal-mean zonal wind (m/s), temperature (K), meridional wind (m/s), `(pfull, lat)` |
| `ps_zm` | (64,) | time+zonal-mean surface pressure, hPa |
| `pk`, `bk` | (26,) | hybrid-sigma half-level coefficients |
| `nmonths` | scalar | months averaged (8) |

## What a correct HS94 climatology looks like (sanity checks)

- **Eddy-driven jets:** westerly maxima of ~**34 m/s** near **±40°**,
  **250 hPa**, symmetric about the equator.
- **Surface winds:** midlatitude westerlies, tropical + polar easterlies.
- **Temperature:** warm tropical near-surface (~305 K), cold poles and a cold
  ~200 K upper level, monotonic equator-to-pole gradient.

This run: jet max **33.6 m/s at 40°, 250 hPa**; max T **305.7 K**. See
`docs/figures/hs_isca_reference.png`.
