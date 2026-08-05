"""Run the Isca Held-Suarez benchmark to generate the jsca validation reference.

This is the pinned Isca ``held_suarez_test_case.py`` (T42L25, dt=600 s,
``damping_order=4``, ``uneven_sigma``) reproduced verbatim except for
sandbox-portability knobs:

  - ``num_cores`` from the ``NCORES`` env var (default 4; the shipped test case
    uses 16). Domain decomposition changes roundoff only, not the climatology.
  - ``mpirun --allow-run-as-root --oversubscribe`` (we run as root in a
    container; see ``baseline/PINNED.md`` deviation #4).
  - number of 30-day months from the ``NMONTHS`` env var (default 12).

Requires the Isca build environment (``$GFDL_BASE``/``$GFDL_ENV`` etc.); it is
NOT a jsca dependency. Feed the ``$GFDL_DATA/held_suarez_bench`` output to
``baseline/make_hs_reference.py`` to build the reference climatology.
"""
import os

from isca import GFDL_BASE, DiagTable, DryCodeBase, Experiment, Namelist

NCORES = int(os.environ.get("NCORES", "4"))
RESOLUTION = 'T42', 25

cb = DryCodeBase.from_directory(GFDL_BASE)

exp_name = 'held_suarez_bench'
exp = Experiment(exp_name, codebase=cb)

diag = DiagTable()
diag.add_file('atmos_monthly', 30, 'days', time_units='days')
diag.add_field('dynamics', 'ps', time_avg=True)
diag.add_field('dynamics', 'bk')
diag.add_field('dynamics', 'pk')
diag.add_field('dynamics', 'ucomp', time_avg=True)
diag.add_field('dynamics', 'vcomp', time_avg=True)
diag.add_field('dynamics', 'temp', time_avg=True)
diag.add_field('dynamics', 'vor', time_avg=True)
diag.add_field('dynamics', 'div', time_avg=True)
exp.diag_table = diag

namelist = Namelist({
    'main_nml': {
        'dt_atmos': 600,
        'days': 30,
        'calendar': 'thirty_day',
        'current_date': [2000, 1, 1, 0, 0, 0],
    },
    'atmosphere_nml': {'idealized_moist_model': False},
    'spectral_dynamics_nml': {
        'damping_order': 4,
        'water_correction_limit': 200.e2,
        'reference_sea_level_press': 1.0e5,
        'valid_range_t': [100., 800.],
        'initial_sphum': 0.0,
        'vert_coord_option': 'uneven_sigma',
        'scale_heights': 6.0,
        'exponent': 7.5,
        'surf_res': 0.5,
    },
    'hs_forcing_nml': {
        't_zero': 315., 't_strat': 200., 'delh': 60., 'delv': 10., 'eps': 0.,
        'sigma_b': 0.7, 'ka': -40., 'ks': -4., 'kf': -1.,
        'do_conserve_energy': True,
    },
    'diag_manager_nml': {'mix_snapshot_average_fields': False},
    'fms_nml': {'domains_stack_size': 600000},
    'fms_io_nml': {'threading_write': 'single', 'fileset_write': 'single'},
})
exp.namelist = namelist
exp.set_resolution(*RESOLUTION)

if __name__ == '__main__':
    nmonths = int(os.environ.get('NMONTHS', '12'))
    mpo = '--allow-run-as-root --oversubscribe'
    exp.run(1, num_cores=NCORES, use_restart=False, mpirun_opts=mpo)
    for i in range(2, nmonths + 1):
        exp.run(i, num_cores=NCORES, mpirun_opts=mpo)
    print('RUN_DONE months=%d' % nmonths)
