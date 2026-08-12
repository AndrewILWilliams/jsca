# Validation

Fidelity is `jsca`'s product, so validation is layered:

- **Tier-1 — module fixtures.** Every ported routine is checked against golden
  input/output data dumped from the instrumented Fortran, at `rtol` from `1e-14`
  down to `1e-11`. These are the {doc}`../gallery/index` component figures.
- **Tier-2 — the physics chain.** The single-column model steps the whole
  physics stack with the dynamical core bypassed, isolating physics fidelity from
  spectral-dynamics chaos.
- **Tier-3 — the climatology.** An ensemble-mean equivalence test (Student-t
  against the Fortran's month-to-month spread, FDR-controlled at 5 %, with
  practical-significance floors) decides whether two multi-year climatologies are
  statistically indistinguishable.

The pages below are the detailed validation records for each configuration.

```{list-table}
:header-rows: 1
:widths: 34 66

* - Configuration
  - Headline result vs Isca
* - Held–Suarez (dry)
  - 0.0 % of (level, lat) points differ beyond internal variability; jet 33.7 vs
    33.6 m/s.
* - Frierson (moist aquaplanet)
  - $q$/precip/$T$/`t_surf` correlations 0.995–0.9996; jet 38.0 vs 38.1 m/s;
    Tier-3 `fail_fraction = 0.0 %` at T21.
* - Single-column model
  - day-40 SST RMSD 0.012 K; $T$ profile 0.13 K; $q$ profile 0.11 g/kg.
* - Bucket (land hydrology)
  - T21 pattern correlations 0.96–1.00 (`bucket_depth`, precip, `t_surf`).
* - +2 % solar response
  - global-mean warming +0.81 K (jsca) vs +0.88 K (Isca); matching
    wet-get-wetter structure.
```

```{toctree}
:maxdepth: 1
:caption: Detailed records

/held_suarez_validation_results
/frierson_climatology
/frierson_solar_response
/single_column_model
/bucket_model
```

## The equivalence test

The Tier-3 decision is not "are the means close" but "is the jsca–Isca difference
smaller than the Fortran's own sampling noise". At each (level, latitude) point
the test forms jsca's ensemble-mean anomaly relative to Isca's ensemble and asks,
via a two-sided Student-t with the Fortran's month-to-month variance, whether it
is distinguishable from zero — controlling the false-discovery rate across all
points and applying a practical floor (e.g. 2 m/s for winds, 1.5 K for
temperature) so sub-noise differences are never flagged. A `fail_fraction` of
0.0 % means no point is distinguishable from Isca beyond its internal
variability. The implementation is {py:mod}`jsca.testing.equivalence`.
