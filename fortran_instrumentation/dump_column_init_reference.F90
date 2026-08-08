!! Standalone driver: golden fixtures for the single-column model's cold-start
!! initial condition, produced by the actual Isca Fortran
!! (src/atmos_column/column_initialize_fields.F90 compiled UNMODIFIED from the
!! pinned tree). This is the one genuinely-new arithmetic routine in the SCM port
!! (jsca/model/column.py::initial_state); the rest of the SCM is an assembly of
!! already-fixtured kernels (leapfrog, press_and_geopot, idealized_moist_phys).
!!
!! column_initialize_fields sets, for a column of num_levels:
!!   ug(:,:,K) = vg(:,:,K) = surface_wind/sqrt(2);  ug/vg = 0 aloft
!!   tg        = initial_temperature
!!   psg       = exp(ln(reference_sea_level_press) - surf_geopotential/(rdgas*T0))
!!
!! Two cases are dumped: zero surface geopotential (psg = p_ref) and a non-zero one
!! (exercises the hydrostatic ps term).
!!
!! Build (from repo root; ISCA_SRC = pinned Isca checkout at commit
!! a290bc376d84d0ee83adbb80eb374b9f629c3534):
!!   cd fortran_instrumentation
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 column_stubs.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_column/column_initialize_fields.F90 \
!!     dump_column_init_reference.F90 -o dump_column_init_reference
!! Run:
!!   JSCA_DUMP_DIR=../tests/fixtures/raw_column_init ./dump_column_init_reference
!! Convert with read_dumps.py into tests/fixtures/column_init_reference.npz.
!!
!! STATUS: fixtures NOT yet generated (no gfortran / Isca build in the porting
!! sandbox). Until they exist, jsca/model/column.py::initial_state is covered by
!! the arithmetic assertion in tests/test_column.py
!! (test_initial_condition_matches_fortran) and the full-driver smoke test, exactly
!! as the frierson assembly is (docs/frierson_roadmap.md).

program dump_column_init_reference
use column_initialize_fields_mod, only: column_initialize_fields
use jsca_dump_mod
implicit none

integer, parameter :: nlev = 25
real, parameter :: ref_slp = 101325.0
real, parameter :: t0 = 264.0
real, parameter :: surface_wind = 5.0

call run(0.0,    'flat')     ! surf_geopotential = 0  -> psg = ref_slp
call run(1000.0, 'topo')     ! surf_geopotential /= 0 -> hydrostatic psg

write(*,*) 'column init reference fixtures dumped'

contains

  subroutine run(phi_s, tag)
    real, intent(in) :: phi_s
    character(len=*), intent(in) :: tag
    real, dimension(1,1) :: psg, surf_geopotential
    real, dimension(1,1,nlev) :: ug, vg, tg
    real :: meta(4)

    surf_geopotential = phi_s
    call column_initialize_fields(ref_slp, t0, surface_wind, &
                                  surf_geopotential, psg, ug, vg, tg)

    meta = (/ ref_slp, t0, surface_wind, phi_s /)
    call jsca_dump_1d('ci_'//trim(tag)//'_meta', meta)
    call jsca_dump_2d('ci_'//trim(tag)//'_psg', psg)
    call jsca_dump_3d('ci_'//trim(tag)//'_ug',  ug)
    call jsca_dump_3d('ci_'//trim(tag)//'_vg',  vg)
    call jsca_dump_3d('ci_'//trim(tag)//'_tg',  tg)
  end subroutine run

end program dump_column_init_reference
