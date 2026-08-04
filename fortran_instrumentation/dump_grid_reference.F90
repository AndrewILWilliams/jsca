!! Standalone driver: dumps Isca's own Gaussian nodes/weights and normalized
!! associated Legendre functions at T42 / 64 latitudes — the first Tier-1
!! golden fixtures, produced by the actual Fortran (compiled against the real
!! gauss_and_legendre.F90 from the Isca tree, with fms_stubs.F90 for logging).
!!
!! Build (from the jsca repo root, ISCA_SRC pointing at the Isca checkout):
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fortran_instrumentation/fms_stubs.F90 \
!!     fortran_instrumentation/jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_spectral/tools/gauss_and_legendre.F90 \
!!     fortran_instrumentation/dump_grid_reference.F90 -o dump_grid_reference
!! Run:
!!   JSCA_DUMP_DIR=tests/fixtures/raw ./dump_grid_reference

program dump_grid_reference
use gauss_and_legendre_mod, only: compute_gaussian, compute_legendre
use jsca_dump_mod, only: jsca_dump_1d, jsca_dump_3d
implicit none

integer, parameter :: n_hem = 32, nlat = 64, nf = 42, ns = 43
real :: sin_hem(n_hem), wts_hem(n_hem)
real :: sin_lat(nlat)
real :: legendre(0:nf, 0:ns, nlat)
integer :: j

call compute_gaussian(sin_hem, wts_hem, n_hem)
call jsca_dump_1d('sin_hem', sin_hem)
call jsca_dump_1d('wts_hem', wts_hem)

! assemble global south -> north exactly as jsca.grid.gaussian_grid does
do j = 1, n_hem
  sin_lat(j) = -sin_hem(j)
  sin_lat(n_hem + j) = sin_hem(n_hem + 1 - j)
end do
call jsca_dump_1d('sin_lat_global', sin_lat)

call compute_legendre(legendre, nf, 1, ns, sin_lat, nlat)
call jsca_dump_3d('legendre_t42', legendre)

write(*,*) 'dumped: sin_hem, wts_hem, sin_lat_global, legendre_t42'
end program dump_grid_reference
