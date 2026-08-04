# Fortran instrumentation — harvesting Tier-1 golden fixtures

The port is validated module-by-module against *instrumented* runs of the
pinned Fortran baseline (scoping doc §4.3). This directory contains the
harvester.

## Files

- `jsca_dump.F90` — dependency-free Fortran module writing float64 stream
  dumps, activated by the `JSCA_DUMP_DIR` environment variable (no-op
  otherwise, so instrumented builds are safe for production runs).
- `read_dumps.py` — numpy reader (Fortran axis order preserved).
- `dump_grid_reference.F90` — standalone driver that links against Isca's own
  `gauss_and_legendre.F90` and dumps Gaussian nodes/weights and normalized
  associated Legendre functions at T42/64-lat: the first real fixtures.

## Recipe

1. Copy `jsca_dump.F90` into the Isca tree (e.g. `src/shared/jsca_dump/`) and
   add its path to `src/path_names` (or the codebase's path list in
   `src/extra/python/isca/codebase.py`).
2. In the routine to fixture, add at entry/exit, e.g. in
   `src/atmos_param/two_stream_gray_rad/two_stream_gray_rad.F90`:

   ```fortran
   use jsca_dump_mod, only: jsca_dump_3d, jsca_dump_2d
   ...
   call jsca_dump_3d('tsgr_in_t',  t)        ! inputs, at entry
   call jsca_dump_2d('tsgr_in_ts', t_surf)
   ...
   call jsca_dump_3d('tsgr_out_tdt', tdt_rad)  ! outputs, before return
   ```

3. Rebuild, run the case **single-PE** for a few steps from a spun-up restart
   with `JSCA_DUMP_DIR=/path/to/dumps`.
4. Convert with `read_dumps.py`, commit compressed fixtures (`.npz`) under
   `tests/fixtures/`, and write the pytest that feeds the dumped inputs to the
   jsca port and compares outputs (tolerances per scoping doc §4.3).

## Ordering caveat

Fixtures pair by call order per name (counter in the filename). Dump inputs
and outputs with distinct names from the *same* call site so pairing is
unambiguous.
