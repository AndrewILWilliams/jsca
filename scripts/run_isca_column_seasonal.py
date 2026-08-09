"""Run Isca's single-column model with do_seasonal=.true. (seasonal + diurnal
insolation) for the jsca do_seasonal SCM validation. Adapted from
exp/test_cases/column_test_case/column_test.py (McKim et al. 2024): identical
physics/namelist, but a single short run with DAILY output so we capture the
spin-up trajectory jsca can be compared against.

This is the reference producer for scripts/compare_column_scm.py and issue #43.

Building Isca in a fresh sandbox (verified 2026-08, gfortran 13.3):

    apt-get install -y gfortran libnetcdf-dev libnetcdff-dev libopenmpi-dev openmpi-bin
    git clone https://github.com/ExeClim/Isca /tmp/isca
    cd /tmp/isca && git checkout a290bc376d84d0ee83adbb80eb374b9f629c3534  # pinned
    # modern gfortran needs these on the old FMS; add to the mkmf template FFLAGS:
    #   -fallow-argument-mismatch -fallow-invalid-boz
    #   (src/extra/python/isca/templates/mkmf.template.gfort)
    pip install f90nml jinja2 sh

Then run this script. NB: Isca's run.sh calls `mpirun -np 1`, which refuses to run
as root and fails silently; if that happens, run the built executable directly in
the run dir instead:
    cd $GFDL_WORK/experiment/scm_reference/run
    ulimit -s unlimited && MALLOC_CHECK_=0 ./column_isca.x > model.log 2>&1
The single-fileset diag output `atmos_daily.nc` lands in that run dir.

Usage: python scripts/run_isca_column_reference.py [run_days]
"""
import os
import sys

os.environ.setdefault("GFDL_BASE", "/tmp/isca")
os.environ.setdefault("GFDL_ENV", "gfortran")
os.environ.setdefault("GFDL_WORK", "/tmp/isca_work")
os.environ.setdefault("GFDL_DATA", "/tmp/isca_data")
sys.path.insert(0, "/tmp/isca/src/extra/python")
import numpy as np
from isca import GFDL_BASE, ColumnCodeBase, DiagTable, Experiment, Namelist

NCORES = 1
RUN_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 90

cb = ColumnCodeBase.from_directory(GFDL_BASE)
cb.compile()

exp = Experiment('scm_seasonal', codebase=cb)

diag = DiagTable()
diag.add_file('atmos_daily', 1, 'days', time_units='days')
diag.add_field('column', 'ps', time_avg=True)
diag.add_field('column', 'bk')
diag.add_field('column', 'pk')
diag.add_field('column', 'temp', time_avg=True)
diag.add_field('column', 'sphum', time_avg=True)
diag.add_field('column', 'ucomp', time_avg=True)
diag.add_field('column', 'vcomp', time_avg=True)
diag.add_field('atmosphere', 'precipitation', time_avg=True)
diag.add_field('mixed_layer', 't_surf', time_avg=True)
diag.add_field('mixed_layer', 'flux_lhe', time_avg=True)
diag.add_field('two_stream', 'swdn_toa', time_avg=True)
exp.diag_table = diag
exp.clear_rundir()

exp.namelist = Namelist({
    'main_nml': {'days': RUN_DAYS, 'hours': 0, 'minutes': 0, 'seconds': 0,
                 'dt_atmos': 1440, 'current_date': [1,1,1,0,0,0], 'calendar': 'thirty_day'},
    'atmosphere_nml': {'idealized_moist_model': True},
    'column_nml': {'lon_max': 1, 'lat_max': 1, 'num_levels': 31,
                   'initial_sphum': 1e-3, 'q_decrease_only': True},
    'column_grid_nml': {'lat_value': float(np.rad2deg(np.arcsin(1/np.sqrt(3))))},
    'column_init_cond_nml': {'initial_temperature': 264., 'surf_geopotential': 0.0,
                             'surface_wind': 5.},
    'idealized_moist_phys_nml': {'do_damping': False, 'turb': True, 'mixed_layer_bc': True,
        'do_simple': True, 'roughness_mom': 3.21e-05, 'roughness_heat': 3.21e-05,
        'roughness_moist': 3.21e-05, 'two_stream_gray': True,
        'convection_scheme': 'SIMPLE_BETTS_MILLER', 'do_lcl_diffusivity_depth': True},
    'two_stream_gray_rad_nml': {'rad_scheme': 'frierson', 'do_seasonal': True, 'atm_abs': 0.2},
    'qe_moist_convection_nml': {'rhbm': 0.7, 'Tmin': 160., 'Tmax': 350.},
    'lscale_cond_nml': {'do_simple': True, 'do_evap': False},
    'surface_flux_nml': {'use_virtual_temp': True, 'do_simple': True, 'old_dtaudv': True},
    'vert_turb_driver_nml': {'do_mellor_yamada': False, 'do_diffusivity': True,
        'do_simple': True, 'constant_gust': 0.0, 'use_tau': False},
    'mixed_layer_nml': {'tconst': 285., 'prescribe_initial_dist': False,
        'evaporation': True, 'depth': 2.5, 'albedo_value': 0.30},
    'sat_vapor_pres_nml': {'do_simple': True},
    'vert_coordinate_nml': {
        'bk': [0.000000, 0.0117665, 0.0196679, 0.0315244, 0.0485411, 0.0719344, 0.1027829,
               0.1418581, 0.1894648, 0.2453219, 0.3085103, 0.3775033, 0.4502789, 0.5244989,
               0.5977253, 0.6676441, 0.7322627, 0.7900587, 0.8400683, 0.8819111, 0.9157609,
               0.9422770, 0.9625127, 0.9778177, 0.9897489, 1.0000000],
        'pk': [0.0]*26},
    'diag_manager_nml': {'mix_snapshot_average_fields': False},
    'fms_nml': {'domains_stack_size': 600000},
    'fms_io_nml': {'threading_write': 'single', 'fileset_write': 'single'},
})

exp.run(1, use_restart=False, num_cores=NCORES)
print("RUN_OK datadir:", exp.datadir)
