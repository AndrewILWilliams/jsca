!! Golden-fixture driver for Isca's saturation vapor pressure (do_simple path).
!!
!! Calls the real, unmodified kernel sat_vapor_pres_k.F90 (which is
!! dependency-free) directly, bypassing the namelist wrapper: sat_vapor_pres_init_k
!! takes do_simple as an argument. Dumps es(T), qs(T,p) and dqs/dT(T) over a
!! physical T,p grid for tests/test_sat_vapor_pres_fixtures.py.
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     $ISCA_SRC/src/shared/sat_vapor_pres/sat_vapor_pres_k.F90 \
!!     jsca_dump.F90 dump_sat_vapor_pres_reference.F90 \
!!     -o dump_sat_vapor_pres_reference
!! Run with JSCA_DUMP_DIR set, then convert with read_dumps.py.

program dump_sat_vapor_pres_reference
  use sat_vapor_pres_k_mod, only: sat_vapor_pres_init_k, lookup_es_k, compute_qs_k
  use jsca_dump_mod, only: jsca_dump_1d
  implicit none

  integer, parameter :: n = 300
  real :: temp(n), press(n), es(n), qs(n), dqs(n)
  real :: eps, zvir, TFREEZE, HLV, RVGAS, RDGAS, ES0
  integer :: i, nbad, nsize
  character(len=256) :: err

  ! constants (match jsca.constants / Isca constants.F90)
  TFREEZE = 273.16;  HLV = 2.500e6;  RVGAS = 461.50;  RDGAS = 287.04;  ES0 = 1.0
  eps = RDGAS / RVGAS;  zvir = 0.0

  ! do_simple table: tcmin_simple=-173, tcmax_simple=350, esres=10
  nsize = (350 - (-173)) * 10 + 1
  call sat_vapor_pres_init_k(nsize, -173.0, 350.0, TFREEZE, HLV, RVGAS, ES0, err, &
                             .false., .true., .false., .false.)

  ! physical T (K) and p (Pa) sweeps
  do i = 1, n
    temp(i)  = 150.0 + (340.0 - 150.0) * real(i - 1) / real(n - 1)
    press(i) = 2000.0 + (1.0e5 - 2000.0) * real(i - 1) / real(n - 1)
  end do

  call lookup_es_k(temp, es, nbad)
  call compute_qs_k(temp, press, eps, zvir, qs, nbad, dqsdT=dqs)

  call jsca_dump_1d('svp_temp', temp)
  call jsca_dump_1d('svp_press', press)
  call jsca_dump_1d('svp_es', es)
  call jsca_dump_1d('svp_qs', qs)
  call jsca_dump_1d('svp_dqsdt', dqs)
  write(*,*) 'SVP_DUMP_DONE nbad=', nbad
end program dump_sat_vapor_pres_reference
