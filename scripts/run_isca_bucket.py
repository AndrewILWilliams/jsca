"""Run the pinned Isca bucket-hydrology experiment at T21 with GREY radiation.

This is the Isca half of the Stage-4 jsca-vs-Isca bucket validation. It reproduces
``exp/test_cases/bucket_hydrology/bucket_model_test_case.py`` with two changes:

* **grey radiation** (``two_stream_gray=True``, ``rad_scheme='frierson'``) instead
  of the shipped RRTM — this is what jsca's ``bucket_model`` implements;
* the **T21 realistic-continents** ``land.nc`` from
  ``scripts/make_bucket_land.py`` instead of Isca's shipped square block.

Everything else matches the pinned namelist: 40-level ``uneven_sigma``,
``SIMPLE_BETTS_MILLER`` convection, ``land_option='input'``, ``bucket=True``,
``max_bucket_depth_land=2``, ``init_bucket_depth_land=1``, mixed-layer
``depth=20``, ``albedo_value=0.25``, ``land_h_capacity_prefactor=0.1``,
``land_albedo_prefactor=1.3``, roughness ``2e-4``, ``robert_coeff=0.03``.

Requires the Isca environment (GFDL_BASE etc.) and a compiled IscaCodeBase. Writes
monthly ``atmos_monthly.nc`` with ``bucket_depth``, ``precipitation``, ``t_surf``,
``temp``, ``sphum``, ``ps``.

Usage::

    GFDL_BASE=/tmp/isca GFDL_WORK=/tmp/gfdl_work GFDL_DATA=/tmp/gfdl_data \\
      GFDL_ENV=gfortran PYTHONPATH=/tmp/isca/src/extra/python \\
      python scripts/run_isca_bucket.py --months 12 --ncores 4 --land /path/land.nc
"""
import argparse
import os
import shutil

from isca import DiagTable, Experiment, IscaCodeBase, Namelist


def build_namelist():
    return Namelist({
        "main_nml": {
            "days": 30, "hours": 0, "minutes": 0, "seconds": 0,
            "dt_atmos": 720, "current_date": [1, 1, 1, 0, 0, 0],
            "calendar": "thirty_day",
        },
        "idealized_moist_phys_nml": {
            "do_damping": True, "turb": True, "mixed_layer_bc": True,
            "do_virtual": False, "do_simple": True,
            "roughness_mom": 2.0e-4, "roughness_heat": 2.0e-4, "roughness_moist": 2.0e-4,
            # GREY radiation (the jsca equivalent), NOT RRTM
            "two_stream_gray": True, "do_rrtm_radiation": False,
            "convection_scheme": "SIMPLE_BETTS_MILLER",
            "land_option": "input", "land_file_name": "INPUT/land.nc",
            "bucket": True, "init_bucket_depth_land": 1.0, "max_bucket_depth_land": 2.0,
        },
        "vert_turb_driver_nml": {
            "do_mellor_yamada": False, "do_diffusivity": True, "do_simple": True,
            "constant_gust": 0.0, "use_tau": False,
        },
        "diffusivity_nml": {"do_entrain": False, "do_simple": True},
        "surface_flux_nml": {
            "use_virtual_temp": False, "do_simple": True, "old_dtaudv": True,
        },
        "atmosphere_nml": {"idealized_moist_model": True},
        "mixed_layer_nml": {
            "tconst": 285.0, "prescribe_initial_dist": True, "evaporation": True,
            "depth": 20.0, "land_option": "input",
            "land_h_capacity_prefactor": 0.1, "albedo_value": 0.25,
            "land_albedo_prefactor": 1.3, "do_qflux": False,
        },
        "qe_moist_convection_nml": {"rhbm": 0.7, "Tmin": 160.0, "Tmax": 350.0},
        "lscale_cond_nml": {"do_simple": True, "do_evap": True},
        "sat_vapor_pres_nml": {"do_simple": True},
        "damping_driver_nml": {
            "do_rayleigh": True, "trayfric": -0.5, "sponge_pbottom": 150.0,
            "do_conserve_energy": True,
        },
        # Frierson grey radiation (matches jsca GrayRadParams defaults)
        "two_stream_gray_rad_nml": {
            "rad_scheme": "frierson", "do_seasonal": False, "atm_abs": 0.2,
        },
        "diag_manager_nml": {"mix_snapshot_average_fields": False},
        "fms_nml": {"domains_stack_size": 600000},
        "fms_io_nml": {"threading_write": "single", "fileset_write": "single"},
        "spectral_dynamics_nml": {
            "damping_order": 4, "water_correction_limit": 200.0e2,
            "reference_sea_level_press": 1.0e5, "num_levels": 40,
            "valid_range_t": [100.0, 800.0], "initial_sphum": [2.0e-6],
            "vert_coord_option": "uneven_sigma", "surf_res": 0.2,
            "scale_heights": 11.0, "exponent": 7.0, "robert_coeff": 0.03,
        },
    })


def build_diag_table():
    diag = DiagTable()
    diag.add_file("atmos_monthly", 30, "days", time_units="days")
    diag.add_field("dynamics", "ps", time_avg=True)
    diag.add_field("dynamics", "bk")
    diag.add_field("dynamics", "pk")
    diag.add_field("atmosphere", "precipitation", time_avg=True)
    diag.add_field("atmosphere", "bucket_depth", time_avg=True)
    diag.add_field("mixed_layer", "t_surf", time_avg=True)
    diag.add_field("dynamics", "sphum", time_avg=True)
    diag.add_field("dynamics", "temp", time_avg=True)
    return diag


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--months", type=int, default=12, help="30-day months to run")
    ap.add_argument("--ncores", type=int, default=4)
    ap.add_argument("--land", default=None, help="path to land.nc (T21 continents)")
    ap.add_argument("--name", default="bucket_grey_t21")
    args = ap.parse_args()

    cb = IscaCodeBase.from_directory(os.environ["GFDL_BASE"])
    cb.compile()

    exp = Experiment(args.name, codebase=cb)
    exp.namelist = build_namelist()
    # set_resolution merges lon_max/lat_max/num_fourier/num_spherical into the
    # existing spectral_dynamics_nml, so it MUST run after the namelist is assigned
    # (otherwise the default T42 grid is used and the T21 land.nc mismatches).
    exp.set_resolution("T21", 40)
    exp.diag_table = build_diag_table()

    land = args.land or os.path.join(os.environ["GFDL_BASE"],
                                     "exp/test_cases/bucket_hydrology/input/land.nc")
    exp.inputfiles = [land]
    exp.clear_rundir()

    exp.run(1, use_restart=False, num_cores=args.ncores)
    for i in range(2, args.months + 1):
        exp.run(i, num_cores=args.ncores)

    # Report where the output landed.
    datadir = os.path.join(os.environ["GFDL_DATA"], args.name)
    print(f"ISCA_BUCKET_RUN_DONE months={args.months} datadir={datadir}")
    last = os.path.join(datadir, f"run{args.months:04d}", "atmos_monthly.nc")
    print("last monthly file:", last, "exists" if os.path.exists(last) else "MISSING")
    # copy land.nc used alongside outputs for provenance
    shutil.copy(land, os.path.join(datadir, "land_used.nc")) if os.path.isdir(datadir) else None


if __name__ == "__main__":
    main()
