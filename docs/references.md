# References & bibliography

## The pinned Isca source

All ports and fixtures refer to one commit —
[`ExeClim/Isca@a290bc37`](https://github.com/ExeClim/Isca/commit/a290bc376d84d0ee83adbb80eb374b9f629c3534)
(master, 2026-01-30). Do not port from a different Isca version; the fixture
tolerances assume this exact source.

- **Isca** — Vallis, G. K., et al. (2018): *Isca, v1.0: A framework for the
  global modelling of the atmospheres of Earth and other planets at varying
  levels of complexity.* Geosci. Model Dev. 11, 843–859.
- Isca documentation: <https://execlim.github.io/Isca/>

## Scientific references for the ported schemes

```{list-table}
:header-rows: 1
:widths: 34 66

* - Component
  - Reference
* - Spectral core / semi-implicit
  - Simmons & Burridge (1981), *Mon. Wea. Rev.* 109, 758–766.
* - RAW time filter
  - Williams (2011), *Mon. Wea. Rev.* 139, 1996–2007 (extending Robert 1966,
    Asselin 1972).
* - Held–Suarez benchmark
  - Held & Suarez (1994), *Bull. Amer. Meteor. Soc.* 75, 1825–1830.
* - Grey radiation
  - Frierson, Held & Zurita-Gotor (2006, 2007), *J. Atmos. Sci.*
* - Humidity-dependent grey LW
  - Byrne & O'Gorman (2013), *J. Climate* 26, 4000–4016.
* - Simplified Betts–Miller convection
  - Frierson (2007), *J. Atmos. Sci.* 64, 1959–1976; virtual-temperature CAPE
    from O'Gorman & Schneider (2008).
* - Full Betts–Miller convection
  - Betts & Miller (1986), *Quart. J. Roy. Meteor. Soc.* 112, 693–709.
* - Dry convective adjustment
  - Schneider & Walker (2006), *J. Atmos. Sci.* 63, 1569–1586.
* - Bucket hydrology
  - Manabe (1969), *Mon. Wea. Rev.* 97, 739–774.
* - Single-column model
  - McKim et al. (2024) — the Isca SCM.
* - Exponential-cutoff damping
  - Smith, Boccaletti et al. (2002), *J. Fluid Mech.* 469, 13–48.
```

## Precedents for the JAX rewrite

- **VEROS** — a Fortran ocean model (pyOM2) rewritten in Python/JAX, with
  benchmarks showing JAX ≈ Fortran on CPU.
- **Dinosaur** (Google) and the **JCM** model (Dinosaur + SPEEDY physics in JAX)
  — architectural precedent for a JAX spectral dycore. `jsca` differs in aiming
  for *faithful* reproduction of a specific Fortran model, not a loose match.

## Project documents

The scoping analysis, roadmaps, and phase checklists live in the repository under
`docs/` (`scoping.md`, `frierson_roadmap.md`, `phase0_checklist.md`) and in
`CLAUDE.md`. They are development records rather than user documentation and are
not part of this manual's navigation.
