!! Standalone driver: golden fixtures for compute_vert_coord (the generalized
!! vertical-coordinate a=pk, b=bk coefficients), produced by the actual Isca
!! Fortran (src/atmos_spectral/init/vert_coordinate.F90 compiled unmodified from
!! the pinned tree). fms_mod/constants_mod are fms_stubs.F90 (the namelist-read
!! 'input' path is never exercised, so open_namelist_file/etc. are link-only).
!!
!! Covers the computed options: even_sigma, uneven_sigma, hybrid (N=25), plus the
!! hard-coded mcm (N=14) and v197 (N=18) profiles. Parameters match Isca's
!! spectral_dynamics_nml defaults.
!!
!! Build (from repo root; ISCA_SRC = pinned Isca checkout):
!!   cd fortran_instrumentation
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_spectral/init/vert_coordinate.F90 \
!!     dump_vert_coordinate_reference.F90 -o dump_vert_coordinate_reference
!! Run:
!!   JSCA_DUMP_DIR=../tests/fixtures/raw_vert_coordinate ./dump_vert_coordinate_reference

program dump_vert_coordinate_reference
use vert_coordinate_mod, only: compute_vert_coord
use jsca_dump_mod
implicit none

real, parameter :: scale_heights = 4.0, surf_res = 0.1, exponent = 2.5
real, parameter :: p_press = 0.1, p_sigma = 0.3, reference_press = 101325.0
real :: meta(6)
real, allocatable :: a(:), b(:)

meta = (/ scale_heights, surf_res, exponent, p_press, p_sigma, reference_press /)
call jsca_dump_1d('vc_meta', meta)

call run('even_sigma',   25, 'even')
call run('uneven_sigma', 25, 'uneven')
call run('hybrid',       25, 'hybrid')
call run('mcm',          14, 'mcm')
call run('v197',         18, 'v197')

write(*,*) 'vert_coordinate reference fixtures dumped'

contains

  subroutine run(option, num_levels, tag)
    character(len=*), intent(in) :: option, tag
    integer, intent(in) :: num_levels
    allocate(a(num_levels+1), b(num_levels+1))
    call compute_vert_coord(option, scale_heights, surf_res, exponent, p_press, p_sigma, &
                            reference_press, a, b)
    call jsca_dump_1d('vc_'//trim(tag)//'_a', a)
    call jsca_dump_1d('vc_'//trim(tag)//'_b', b)
    deallocate(a, b)
  end subroutine run

end program dump_vert_coordinate_reference
