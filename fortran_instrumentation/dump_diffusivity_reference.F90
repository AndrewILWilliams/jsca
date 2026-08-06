!! Golden-fixture driver for Isca's boundary-layer diffusivity (diffusivity.F90),
!! the Frierson simple-diffusivity path: diffusivity_nml do_simple=.true.,
!! do_entrain=.false. (defaults: fixed_depth=F, free_atm_diff=F, pbl_mcm=F,
!! background_m/t=0, frac_inner=0.1, rich_crit_pbl=1.0, use_pog_bug_fix=T).
!! This is the K-profile scheme vert_turb_driver calls when do_diffusivity=.true.
!! (with constant_gust=0, so gust=0 and z_pbl=h).
!!
!! Compiles the REAL, unmodified diffusivity.F90 with the real monin_obukhov
!! wrapper (+ _PURE kernel) for mo_diff, against the shared stubs. Namelist
!! injected through FMS's internal-file buffer (build -DINTERNAL_FILE_NML).
!!
!! Dumps k_m (diff_m), k_t (diff_t) and the PBL depth h (z_pbl) over a column set
!! spanning stable/unstable near-surface conditions.
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none -cpp \
!!     -DINTERNAL_FILE_NML -fallow-argument-mismatch \
!!     -I$ISCA_SRC/src/shared/include -I$ISCA_SRC/src/atmos_param/monin_obukhov \
!!     fms_stubs.F90 mpp_mod_stub.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_param/monin_obukhov/monin_obukhov_kernel.F90 \
!!     $ISCA_SRC/src/atmos_param/monin_obukhov/monin_obukhov.F90 \
!!     $ISCA_SRC/src/atmos_param/diffusivity/diffusivity.F90 \
!!     dump_diffusivity_reference.F90 -o dump_diffusivity_reference
!! Run with JSCA_DUMP_DIR set, then convert with read_dumps.py.

program dump_diffusivity_reference
  use          mpp_mod, only: input_nml_file
  use  diffusivity_mod, only: diffusivity
  use    jsca_dump_mod, only: jsca_dump_2d, jsca_dump_3d

  implicit none

  integer, parameter :: nlon = 3, nlat = 6, nz = 20
  real, dimension(nlon,nlat,nz)   :: t, q, u, v, p_full, z_full, k_m, k_t
  real, dimension(nlon,nlat,nz+1) :: p_half, z_half
  real, dimension(nlon,nlat)      :: u_star, b_star, h
  real :: sig_e(nz+1), sig_f(nz), ps, tsurf, lapse, H0, spd
  integer :: i, j, k, unit

  ! diffusivity_init guards its namelist read with file_exist('input.nml') even
  ! on the INTERNAL_FILE_NML path, so a real input.nml must be present; the read
  ! itself still comes from the injected input_nml_file buffer.
  allocate(input_nml_file(2))
  input_nml_file(1) = '&diffusivity_nml do_simple=.true. do_entrain=.false. /'
  input_nml_file(2) = ' '
  open(newunit=unit, file='input.nml', status='replace')
  write(unit,'(a)') '&diffusivity_nml do_simple=.true. do_entrain=.false. /'
  close(unit)

  ps = 1.0e5
  H0 = 7000.0    ! scale height for hydrostatic heights (m)
  q = 1.0e-3

  do k = 1, nz+1
    sig_e(k) = real(k-1) / real(nz)
  end do
  do k = 1, nz
    sig_f(k) = 0.5 * (sig_e(k) + sig_e(k+1))
  end do

  do j = 1, nlat
    do i = 1, nlon
      ! surface-layer state varies across the grid: friction velocity and the
      ! buoyancy scale sweep stable (b_star<0) to unstable (b_star>0).
      u_star(i,j) = 0.10 + 0.30 * real(i-1) / real(nlon-1)
      b_star(i,j) = -0.010 + 0.055 * real(j-1) / real(nlat-1)
      tsurf = 295.0
      lapse = 7.0e-3        ! K/m
      spd = 5.0 + 5.0 * real(i-1)/real(nlon-1)
      do k = 1, nz
        p_full(i,j,k) = sig_f(k) * ps
        p_half(i,j,k) = sig_e(k) * ps
        z_full(i,j,k) = H0 * log(ps / max(p_full(i,j,k), 1.0))
        z_half(i,j,k) = H0 * log(ps / max(sig_e(k) * ps, 1.0))
        t(i,j,k) = tsurf - lapse * z_full(i,j,k)
        t(i,j,k) = max(t(i,j,k), 200.0)
        u(i,j,k) = spd * (1.0 - 0.5 * sig_f(k))    ! shear
        v(i,j,k) = 0.0
      end do
      p_half(i,j,nz+1) = ps
      z_half(i,j,nz+1) = 0.0    ! surface
    end do
  end do

  k_m = 0.0
  k_t = 0.0
  call diffusivity(t, q, u, v, p_full, p_half, z_full, z_half, &
                   u_star, b_star, h, k_m, k_t)

  call jsca_dump_3d('df_t',      t)
  call jsca_dump_3d('df_u',      u)
  call jsca_dump_3d('df_v',      v)
  call jsca_dump_3d('df_p_full', p_full)
  call jsca_dump_3d('df_p_half', p_half)
  call jsca_dump_3d('df_z_full', z_full)
  call jsca_dump_3d('df_z_half', z_half)
  call jsca_dump_2d('df_u_star', u_star)
  call jsca_dump_2d('df_b_star', b_star)
  call jsca_dump_3d('df_k_m',    k_m)
  call jsca_dump_3d('df_k_t',    k_t)
  call jsca_dump_2d('df_h',      h)

  write(*,*) 'DIFF_DUMP_DONE  k_m max:', maxval(k_m), '  h min/max:', minval(h), maxval(h)
end program dump_diffusivity_reference
