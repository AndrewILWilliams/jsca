# Frierson column-physics step fixture — Isca instrumentation recipe

Golden reference for `tests/test_idealized_moist_phys_fixtures.py`: one real step of
the pinned Isca `frierson_test_case` at T42 L25, dumping the `idealized_moist_phys`
column-physics I/O. Unlike the per-module fixtures (which compile a single routine
against stubs), this needs the **full Isca model built and run**, because
`idealized_moist_phys` is the driver that couples every column scheme.

This is what upgrades the assembly's stability smoke test to a machine-precision
fidelity gate, and it is the same machinery item 11c uses for the climatology.

## Prerequisites (available in the sandbox)

`gfortran`, `mpif90`/`mpirun`, `libnetcdf-dev`/`libnetcdff-dev`, `libopenmpi-dev`.
The Isca Python front-end is pure-Python; add it to `PYTHONPATH` (its `setup.py`
is too old for modern setuptools, so don't `pip install -e` it):

```bash
export ISCA_SRC=/path/to/Isca            # the pinned checkout (a290bc3)
export GFDL_BASE=$ISCA_SRC GFDL_ENV=gfortran
export GFDL_WORK=/tmp/isca_work GFDL_DATA=/tmp/isca_data
export PYTHONPATH=$ISCA_SRC/src/extra/python:$PYTHONPATH
pip install f90nml jinja2 sh   # front-end deps
```

The `gfortran` env file (`$ISCA_SRC/src/extra/env/gfortran`) selects the bundled
`mkmf.template.gfort`, which already carries `-fallow-argument-mismatch`
(required for the old FMS against modern gfortran).

## Instrumentation

`jsca_dump.F90` (from this directory, copied into
`$ISCA_SRC/src/atmos_spectral/model/`) provides `jsca_dump_mod`, active only when
`JSCA_DUMP_DIR` is set. Add it to the moist codebase's build list once:

```bash
echo "atmos_spectral/model/jsca_dump.F90" >> $ISCA_SRC/src/extra/model/isca/path_names
```

In `$ISCA_SRC/src/atmos_spectral/driver/solo/idealized_moist_phys.F90`:

- `use jsca_dump_mod, only: jsca_dump_scalar, jsca_dump_2d, jsca_dump_3d`
- a module logical `jsca_imp_first = .true.` (dump the first step only);
- at the **start** of `subroutine idealized_moist_phys` (just after `delta_t` is
  set), dump the inputs: `rad_lat`, `rad_lon`, `ug/vg/tg(:,:,:,previous)`,
  `grid_tracers(:,:,:,previous,nsphum)`, `p_half/p_full(:,:,:,previous)` and
  `(:,:,:,current)`, `z_full(:,:,:,current)`, `z_surf`, `t_surf`, `delta_t`,
  `dt_real`, and the entering `dt_ug/dt_vg/dt_tg/dt_tracers(:,:,:,nsphum)`
  (prefix `imp_*`, `_in` for the tendencies);
- just before `non_diff_dt_ug = dt_ug`, dump the diffusion-chain intermediates:
  `z_half(:,:,:,current)`, `diff_m`, `diff_t`, `flux_t/q/u/v`, `ustar`, `bstar`,
  `dtaudu_atm`, `net_surf_sw_down`, `surf_lw_down`, and the pre-diffusion
  `dt_*_nondiff`;
- at the **end** of the subroutine, dump `dt_ug/dt_vg/dt_tg/dt_tracers(nsphum)`
  (`_out`) and `t_surf` (`imp_tsurf_out`), then set `jsca_imp_first = .false.`.

The jsca driver returns physics-only tendencies, so the test uses Isca's
`(out - in)`.

## Build & run (2 steps is enough; the dump fires on step 1)

```bash
python -c "from isca import IscaCodeBase, GFDL_BASE; IscaCodeBase.from_directory(GFDL_BASE).compile()"
# set up a T42 run from frierson_test_case.py (num_fourier=42, lon_max=128,
# lat_max=64, num_levels=25, dt_atmos=720, the Frierson namelist), then in the run
# directory with the freshly-built isca.x:
source $ISCA_SRC/src/extra/env/gfortran
ulimit -s unlimited; export MALLOC_CHECK_=0
JSCA_DUMP_DIR=/tmp/frierson_dump mpirun --allow-run-as-root -np 1 ./isca.x
```

Convert with `read_dumps.read_all` (take the first entry of each `imp_*` list =
step 1) and subsample columns (column physics is pointwise) into
`tests/fixtures/idealized_moist_phys_reference.npz`.

## Two bugs this caught (both invisible to the stability smoke test)

1. **`z_half`.** The boundary-layer `diffusivity` needs the real geopotential
   half-level heights; a midpoint interpolation of `z_full` is off by tens of km
   near the model top and corrupts the diffusion tendency (~30% error). Fixed by
   threading the real `z_half` (from `compute_geopotential`) through the driver.
2. **Gustiness.** `surface_flux` runs before `vert_turb_driver`, so on step 1 it
   uses Isca's initialised `gust = 1.0` m/s. Passing `gust = 0` makes `w_atm` too
   small — `u_star` ~1.7x low, `b_star` ~10x high — cascading through the whole
   diffusion chain. Fixed by threading `gust` as an input.

With both fixed, the jsca column driver matches Isca to machine precision on
momentum and to the documented `sat_vapor_pres` deviation (~1e-9) on T/q.

## Extending to the climatology (item 11c)

Same build; run the full `frierson_test_case` (30-day months, a multi-year
spin-up + averaging period) with `JSCA_DUMP_DIR` **unset** (so `jsca_dump` is
inert). The monthly `atmos_monthly.nc` zonal-mean climatology is the reference for
the `jsca.testing` statistical-parity comparison against a matching jsca
`FriersonModel` run.
