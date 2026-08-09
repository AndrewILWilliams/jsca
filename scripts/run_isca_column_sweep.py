"""Run Isca's single-column model across a latitude sweep, for the jsca-vs-Isca
parameter-sweep validation (issue #43). Same physics/namelist as
run_isca_column_reference.py (canonical column_test.py), varying only
column_grid_nml:lat_value.

Each latitude sees a different Frierson p2 insolation, so this exercises the physics
chain across regimes (warm/wet tropics -> cold/dry high latitudes) rather than the
single global-average column.

Writes one NetCDF per latitude into baseline/reference/, then
scripts/distill_column_sweep.py reduces them to a single committed .npz.

Usage: python scripts/run_isca_column_sweep.py [run_days]   (default 40)
Prereq: scripts/build_isca_column.sh (toolchain + pinned Isca).
"""
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("GFDL_BASE", "/tmp/isca")
os.environ.setdefault("GFDL_ENV", "gfortran")
os.environ.setdefault("GFDL_WORK", "/tmp/isca_work")
os.environ.setdefault("GFDL_DATA", "/tmp/isca_data")
# OpenMPI refuses to run as root without these (Isca's run.sh calls `mpirun -np 1`).
os.environ.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
os.environ.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
sys.path.insert(0, "/tmp/isca/src/extra/python")
from isca import GFDL_BASE, ColumnCodeBase, DiagTable, Experiment, Namelist

RUN_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
LATS = [0.0, 15.0, 30.0, 45.0, 60.0]
OUT = Path(__file__).resolve().parent.parent / "baseline" / "reference"

cb = ColumnCodeBase.from_directory(GFDL_BASE)
cb.compile()

BK = [0.000000, 0.0117665, 0.0196679, 0.0315244, 0.0485411, 0.0719344, 0.1027829,
      0.1418581, 0.1894648, 0.2453219, 0.3085103, 0.3775033, 0.4502789, 0.5244989,
      0.5977253, 0.6676441, 0.7322627, 0.7900587, 0.8400683, 0.8819111, 0.9157609,
      0.9422770, 0.9625127, 0.9778177, 0.9897489, 1.0000000]


def namelist_for(lat):
    return Namelist({
        'main_nml': {'days': RUN_DAYS, 'hours': 0, 'minutes': 0, 'seconds': 0,
                     'dt_atmos': 1440, 'current_date': [1,1,1,0,0,0], 'calendar': 'thirty_day'},
        'atmosphere_nml': {'idealized_moist_model': True},
        'column_nml': {'lon_max': 1, 'lat_max': 1, 'num_levels': 31,
                       'initial_sphum': 1e-3, 'q_decrease_only': True},
        'column_grid_nml': {'lat_value': float(lat)},
        'column_init_cond_nml': {'initial_temperature': 264., 'surf_geopotential': 0.0,
                                 'surface_wind': 5.},
        'idealized_moist_phys_nml': {'do_damping': False, 'turb': True, 'mixed_layer_bc': True,
            'do_simple': True, 'roughness_mom': 3.21e-05, 'roughness_heat': 3.21e-05,
            'roughness_moist': 3.21e-05, 'two_stream_gray': True,
            'convection_scheme': 'SIMPLE_BETTS_MILLER', 'do_lcl_diffusivity_depth': True},
        'two_stream_gray_rad_nml': {'rad_scheme': 'frierson', 'do_seasonal': False, 'atm_abs': 0.2},
        'qe_moist_convection_nml': {'rhbm': 0.7, 'Tmin': 160., 'Tmax': 350.},
        'lscale_cond_nml': {'do_simple': True, 'do_evap': False},
        'surface_flux_nml': {'use_virtual_temp': True, 'do_simple': True, 'old_dtaudv': True},
        'vert_turb_driver_nml': {'do_mellor_yamada': False, 'do_diffusivity': True,
            'do_simple': True, 'constant_gust': 0.0, 'use_tau': False},
        'mixed_layer_nml': {'tconst': 285., 'prescribe_initial_dist': False,
            'evaporation': True, 'depth': 2.5, 'albedo_value': 0.30},
        'sat_vapor_pres_nml': {'do_simple': True},
        'vert_coordinate_nml': {'bk': BK, 'pk': [0.0]*26},
        'diag_manager_nml': {'mix_snapshot_average_fields': False},
        'fms_nml': {'domains_stack_size': 600000},
        'fms_io_nml': {'threading_write': 'single', 'fileset_write': 'single'},
    })


def diag_table():
    d = DiagTable()
    d.add_file('atmos_daily', 1, 'days', time_units='days')
    for f in ('ps', 'bk', 'pk', 'temp', 'sphum', 'ucomp', 'vcomp'):
        d.add_field('column', f, time_avg=(f not in ('bk', 'pk')))
    d.add_field('atmosphere', 'precipitation', time_avg=True)
    d.add_field('mixed_layer', 't_surf', time_avg=True)
    return d


for lat in LATS:
    exp = Experiment(f'scm_sweep_lat{int(lat)}', codebase=cb)
    exp.diag_table = diag_table()
    exp.clear_rundir()
    exp.namelist = namelist_for(lat)
    exp.run(1, use_restart=False, num_cores=1)
    src = Path(exp.datadir) / 'run0001' / 'atmos_daily.nc'
    if not src.exists():                       # fall back to the rundir copy
        src = Path(os.environ['GFDL_WORK']) / 'experiment' / exp.name / 'run' / 'atmos_daily.nc'
    dst = OUT / f'column_scm_isca_lat{int(lat)}.nc'
    shutil.copy(src, dst)
    print(f'lat={lat}: wrote {dst}')

print('sweep complete:', [int(x) for x in LATS])
