!! Golden-fixture driver for Isca's large-scale condensation (lscale_cond.F90),
!! Frierson options: do_simple=.true., do_evap=.true., hc=1.0.
!!
!! Compiles the REAL, unmodified lscale_cond.F90 and the REAL sat_vapor_pres.F90
!! wrapper (+ kernel) against the shared fms/mpp/constants stubs. The Frierson
!! namelist is injected through FMS's internal-file buffer input_nml_file (build
!! with -DINTERNAL_FILE_NML) so the modules pick up do_simple/do_evap/hc without
!! an input.nml on disk.
!!
!! The input column set is built physically: temperature/pressure profiles, then
!! qsat from the real compute_qs, then q = RH*qsat with an RH profile that makes
!! the upper-mid layers supersaturated (condensation) and the near-surface layers
!! dry (so falling rain re-evaporates). Dumps tin, qin, pfull, phalf and the
!! outputs tdel, qdel, rain for tests/test_lscale_cond_fixtures.py.
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     -DINTERNAL_FILE_NML \
!!     fms_stubs.F90 mpp_mod_stub.F90 mpp_io_stub.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/shared/sat_vapor_pres/sat_vapor_pres_k.F90 \
!!     $ISCA_SRC/src/shared/sat_vapor_pres/sat_vapor_pres.F90 \
!!     $ISCA_SRC/src/atmos_param/lscale_cond/lscale_cond.F90 \
!!     dump_lscale_cond_reference.F90 -o dump_lscale_cond_reference
!! Run with JSCA_DUMP_DIR set, then convert with read_dumps.py.

program dump_lscale_cond_reference
  use            mpp_mod, only: input_nml_file
  use sat_vapor_pres_mod, only: compute_qs, sat_vapor_pres_init
  use     lscale_cond_mod, only: lscale_cond, lscale_cond_init
  use       jsca_dump_mod, only: jsca_dump_2d, jsca_dump_3d
  implicit none

  integer, parameter :: ni = 6, nj = 5, kx = 20
  real, dimension(ni,nj,kx)   :: tin, qin, pfull, qsat, rh, tdel, qdel
  real, dimension(ni,nj,kx+1) :: phalf
  real, dimension(ni,nj)      :: rain, snow, ps
  logical, dimension(ni,nj)   :: coldT
  real    :: sig_e(kx+1), sig_f(kx), tsurf, ttop, lat
  integer :: i, j, k

  ! --- inject the Frierson namelists into FMS's internal-file buffer ---
  allocate(input_nml_file(4))
  input_nml_file(1) = '&sat_vapor_pres_nml do_simple=.true. /'
  input_nml_file(2) = '&lscale_cond_nml do_simple=.true. do_evap=.true. hc=1.0 /'
  input_nml_file(3) = ' '
  input_nml_file(4) = ' '

  call sat_vapor_pres_init()
  call lscale_cond_init()

  ! --- sigma half-levels (top=0 -> surface=1), full-levels at midpoints ---
  do k = 1, kx+1
    sig_e(k) = real(k-1) / real(kx)
  end do
  do k = 1, kx
    sig_f(k) = 0.5 * (sig_e(k) + sig_e(k+1))
  end do

  coldT = .false.
  do j = 1, nj
    lat = -60.0 + 120.0 * real(j-1) / real(nj-1)        ! -60 .. +60 deg
    do i = 1, ni
      ! surface pressure with mild horizontal variation
      ps(i,j) = 1.0e5 + 3.0e3 * cos(real(i-1)/real(ni)*6.2831853) &
                       - 2.0e3 * sin(lat*3.14159/180.0)
      ! temperature: warm surface, cold top, weakened toward the pole
      tsurf = 300.0 - 25.0 * (sin(lat*3.14159/180.0))**2
      ttop  = 205.0
      do k = 1, kx
        phalf(i,j,k) = sig_e(k) * ps(i,j)
        tin(i,j,k)   = ttop + (tsurf - ttop) * sig_f(k)**1.2
      end do
      phalf(i,j,kx+1) = ps(i,j)
      do k = 1, kx
        pfull(i,j,k) = 0.5 * (phalf(i,j,k) + phalf(i,j,k+1))
      end do
    end do
  end do
  ! guard the very top full level (sigma 0 gives zero pressure)
  where (pfull < 100.0) pfull = 100.0

  ! saturation humidity from the REAL compute_qs (do_simple, hc=1)
  call compute_qs(tin, pfull, qsat)

  ! RH profile designed to exercise every branch AND leave surviving rain:
  !   k=5..8   supersaturated upper band  (cold, small condensate)
  !   k=10     dry band                   (re-evaporates the upper rain)
  !   k=15..18 supersaturated lower band  (warm, large condensate)
  !   k=19..20 near-saturated             (little re-evap -> lower rain survives)
  !   elsewhere moderately dry.
  ! Cold upper condensate is small and mostly re-evaporates in the warm dry band;
  ! the warm low-level condensate is large with only near-saturated layers below,
  ! so it reaches the surface as non-trivial rain.
  do k = 1, kx
    do j = 1, nj
      do i = 1, ni
        if (k >= 5 .and. k <= 8) then
          rh(i,j,k) = 1.12 + 0.15 * cos(real(i-1)/real(ni)*6.2831853)   ! supersat (upper)
        else if (k == 10) then
          rh(i,j,k) = 0.35 + 0.10 * real(j-1)/real(nj-1)                ! dry band
        else if (k >= 15 .and. k <= 18) then
          rh(i,j,k) = 1.08 + 0.10 * sin(real(j-1)/real(nj)*6.2831853)   ! supersat (lower)
        else if (k >= 19) then
          rh(i,j,k) = 0.97                                             ! near-sat
        else
          rh(i,j,k) = 0.55                                             ! moderately dry
        end if
      end do
    end do
  end do
  qin = rh * qsat

  call lscale_cond(tin, qin, pfull, phalf, coldT, rain, snow, tdel, qdel)

  call jsca_dump_3d('lc_tin',   tin)
  call jsca_dump_3d('lc_qin',   qin)
  call jsca_dump_3d('lc_pfull', pfull)
  call jsca_dump_3d('lc_phalf', phalf)
  call jsca_dump_3d('lc_tdel',  tdel)
  call jsca_dump_3d('lc_qdel',  qdel)
  call jsca_dump_2d('lc_rain',  rain)
  write(*,*) 'LSCALE_COND_DUMP_DONE  sum(rain)=', sum(rain), '  sum|qdel|=', sum(abs(qdel))
end program dump_lscale_cond_reference
