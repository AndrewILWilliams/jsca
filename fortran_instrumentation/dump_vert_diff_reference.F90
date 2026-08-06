!! Golden-fixture driver for Isca's implicit vertical diffusion (vert_diff.F90),
!! the Frierson down/up split (gcm_vert_diff_down then gcm_vert_diff_up), with
!! do_conserve_energy=.true., do_virtual=.false. (as hard-coded in
!! idealized_moist_phys). Single prognostic tracer (sphum), no bucket/kbot.
!!
!! This exercises the full tridiagonal machinery: the momentum solve (with the
!! surface stress as the implicit bottom BC and frictional dissipative heating),
!! and the temperature/humidity forward-elimination + back-substitution. The
!! surface T/q coupling that the mixed layer supplies is item 8, so here the
!! up sweep uses the down pass's Tri_surf (no mixed-layer update in between) --
!! a well-defined diffusion with the surface T/q flux held at its stored value.
!!
!! Compiles the REAL, unmodified vert_diff.F90 against fms/mpp stubs and a
!! field/tracer-manager stub advertising one prognostic tracer (sphum).
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 mpp_mod_stub.F90 vert_diff_stubs.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_param/vert_diff/vert_diff.F90 \
!!     dump_vert_diff_reference.F90 -o dump_vert_diff_reference
!! Run with JSCA_DUMP_DIR set, then convert with read_dumps.py.

program dump_vert_diff_reference
  use   vert_diff_mod, only: vert_diff_init, gcm_vert_diff_down, &
                             gcm_vert_diff_up, surf_diff_type
  use    jsca_dump_mod, only: jsca_dump_2d, jsca_dump_3d

  implicit none

  integer, parameter :: nlon = 3, nlat = 5, nz = 20
  real, dimension(nlon,nlat,nz)   :: u, v, t, q, diff_m, diff_t, p_full, z_full, &
       dt_u, dt_v, dt_t, dt_q, diss_heat
  real, dimension(nlon,nlat,nz,1) :: tr, dt_tr
  real, dimension(nlon,nlat,nz+1) :: p_half
  real, dimension(nlon,nlat)      :: tau_u, tau_v, dtau_du, dtau_dv
  type(surf_diff_type) :: Tri_surf
  real :: sig_e(nz+1), sig_f(nz), ps, H0, tsurf, lapse, delt, kmax, zh
  integer :: i, j, k

  ps = 1.0e5;  H0 = 7000.0;  delt = 720.0

  do k = 1, nz+1
    sig_e(k) = real(k-1) / real(nz)
  end do
  do k = 1, nz
    sig_f(k) = 0.5 * (sig_e(k) + sig_e(k+1))
  end do

  call vert_diff_init(Tri_surf, nlon, nlat, nz, .true., .false.)

  do j = 1, nlat
    do i = 1, nlon
      tsurf = 298.0 - 3.0 * real(j-1) / real(nlat-1)
      lapse = 6.5e-3
      kmax  = 5.0 + 10.0 * real(i-1) / real(nlon-1)    ! peak diffusivity (m^2/s)
      tau_u(i,j)   = -0.05 - 0.10 * real(i-1)/real(nlon-1)   ! surface stress (Pa)
      tau_v(i,j)   =  0.02
      dtau_du(i,j) = -0.01
      dtau_dv(i,j) = -0.01
      do k = 1, nz
        p_full(i,j,k) = sig_f(k) * ps
        p_half(i,j,k) = sig_e(k) * ps
        z_full(i,j,k) = H0 * log(ps / max(p_full(i,j,k), 1.0))
        t(i,j,k) = max(tsurf - lapse * z_full(i,j,k), 200.0)
        q(i,j,k) = 1.0e-2 * exp(-z_full(i,j,k) / 3000.0)   ! humidity decaying with height
        u(i,j,k) = 8.0 * (1.0 - 0.5 * sig_f(k))
        v(i,j,k) = 2.0 * sig_f(k)
        ! K-profile: peaks in the lower boundary layer, ~0 aloft
        zh = z_full(i,j,k)
        diff_m(i,j,k) = kmax * exp(-((zh - 400.0)/500.0)**2)
        diff_t(i,j,k) = 0.8 * kmax * exp(-((zh - 350.0)/450.0)**2)
      end do
      p_half(i,j,nz+1) = ps
    end do
  end do

  tr(:,:,:,1) = q
  dt_u = 0.0; dt_v = 0.0; dt_t = 0.0; dt_q = 0.0; dt_tr = 0.0

  ! tau_u/tau_v are intent(inout) -- dump the INPUT surface stress before the
  ! call overwrites them with the post-diffusion stress.
  call jsca_dump_2d('vd_tau_u', tau_u)
  call jsca_dump_2d('vd_tau_v', tau_v)

  call gcm_vert_diff_down(1, 1, delt, u, v, t, q, tr, diff_m, diff_t, &
       p_half, p_full, z_full, tau_u, tau_v, dtau_du, dtau_dv, &
       dt_u, dt_v, dt_t, dt_q, dt_tr, diss_heat, Tri_surf)

  call gcm_vert_diff_up(1, 1, delt, Tri_surf, dt_t, dt_q, dt_tr)

  call jsca_dump_3d('vd_u',       u)
  call jsca_dump_3d('vd_v',       v)
  call jsca_dump_3d('vd_t',       t)
  call jsca_dump_3d('vd_q',       q)
  call jsca_dump_3d('vd_diff_m',  diff_m)
  call jsca_dump_3d('vd_diff_t',  diff_t)
  call jsca_dump_3d('vd_p_full',  p_full)
  call jsca_dump_3d('vd_p_half',  p_half)
  call jsca_dump_3d('vd_z_full',  z_full)
  call jsca_dump_2d('vd_dtau_du', dtau_du)
  call jsca_dump_2d('vd_dtau_dv', dtau_dv)
  call jsca_dump_3d('vd_dt_u',    dt_u)
  call jsca_dump_3d('vd_dt_v',    dt_v)
  call jsca_dump_3d('vd_dt_t',    dt_t)
  call jsca_dump_3d('vd_dt_q',    dt_q)
  call jsca_dump_3d('vd_diss_heat', diss_heat)

  write(*,*) 'VD_DUMP_DONE  dt_u range', minval(dt_u), maxval(dt_u)
  write(*,*) '  dt_t range', minval(dt_t), maxval(dt_t)
end program dump_vert_diff_reference
