# Column convection golden-step fixture — Isca instrumentation recipe

Golden reference for `tests/test_column_convection_step_fixtures.py`: the
`qe_moist_convection` I/O at a **real single-column step** of the pinned Isca
`column_test.py` run (31 even-sigma levels, `rhbm=0.7`), so jsca's convection is
validated at the exact column state and config — not just synthetic per-column
inputs.

This was the decisive test for the jsca-vs-Isca precipitation difference. A
daily-mean-state convection call over-rained by ~6%, which looked like a convection
bug; fed Isca's true instantaneous step, jsca reproduces the rain to +0.00% and the
tendencies to the `sat_vapor_pres` es-deviation floor. So the precip difference is a
downstream consequence of a small boundary-layer profile difference, **not** a
convection-scheme error.

## Prerequisites

A built Isca column model (`scripts/build_isca_column.sh`; `ColumnCodeBase`). Same
`gfortran`/netCDF/OpenMPI toolchain as the SCM reference runs.

## Instrumentation

`jsca_dump.F90` (this directory) provides `jsca_dump_mod`, active only when
`JSCA_DUMP_DIR` is set. Add it to the column build's file list once:

```bash
echo "atmos_spectral/driver/solo/jsca_dump.F90" >> $ISCA_SRC/src/extra/model/column/path_names
cp fortran_instrumentation/jsca_dump.F90 $ISCA_SRC/src/atmos_spectral/driver/solo/
```

In `$ISCA_SRC/src/atmos_spectral/driver/solo/idealized_moist_phys.F90`:

- `use jsca_dump_mod, only: jsca_dump_2d, jsca_dump_3d, jsca_dump_scalar`
- a module counter `integer, save :: jsca_conv_step = 0`;
- immediately **after** the `qe_moist_convection` call (the `SIMPLE_BETTS_CONV`
  case, just before `tg_tmp = conv_dt_tg + tg(:,:,:,previous)`), dump the I/O on one
  spun-up step:

  ```fortran
  jsca_conv_step = jsca_conv_step + 1
  if (jsca_conv_step == 600) then
    call jsca_dump_scalar('qe_dt', delta_t)
    call jsca_dump_3d('qe_tin',   tg(:,:,:,previous))
    call jsca_dump_3d('qe_qin',   grid_tracers(:,:,:,previous,nsphum))
    call jsca_dump_3d('qe_pfull', p_full(:,:,:,previous))
    call jsca_dump_3d('qe_phalf', p_half(:,:,:,previous))
    call jsca_dump_3d('qe_dtg',   conv_dt_tg)
    call jsca_dump_3d('qe_dqg',   conv_dt_qg)
    call jsca_dump_2d('qe_rain',  rain)
    call jsca_dump_2d('qe_klzb',  real(klzbs))
    call jsca_dump_2d('qe_flag',  real(convflag))
  endif
  ```

Adding a module `use` changes the build dependency graph, so **delete the build
Makefile** (`$GFDL_WORK/experiment/.../build/column_isca/Makefile`) before
recompiling so mkmf regenerates it, then rebuild via `ColumnCodeBase.compile()`.

## Run & convert

Run the canonical column config (`scripts/run_isca_column_reference.py`) for ≥11
days (60 steps/day, so step 600 fires) with `JSCA_DUMP_DIR` set; then

```python
from read_dumps import read_all; import numpy as np
d = read_all(dump_dir); g = lambda k: np.asarray(d[k][0]).ravel()
np.savez_compressed('tests/fixtures/column_convection_step_reference.npz',
    **{k: g(k) for k in ('qe_dt','qe_tin','qe_qin','qe_pfull','qe_phalf',
                          'qe_dtg','qe_dqg','qe_rain','qe_klzb','qe_flag')})
```
