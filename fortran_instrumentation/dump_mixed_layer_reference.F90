!! Golden-fixture driver for Isca's slab-ocean surface energy balance
!! (mixed_layer.F90), Frierson settings (mixed_layer_nml: depth=2.5,
!! albedo_value=0.31, evaporation=.true., tconst=285, prescribe_initial_dist),
!! land_option='none' (pure ocean), no q-flux, no SST reading.
!!
!! The mixed_layer step is a closed arithmetic form: it takes the surface fluxes
!! and their derivatives, the downward radiation, and the vert_diff surface
!! coupling terms (Tri_surf), and returns the sea-surface-temperature increment
!! plus the updated lowest-level implicit increments (Tri_surf%delta_t/delta_tr
!! for the vert_diff up sweep).
!!
!! Compiles the REAL, unmodified mixed_layer.F90 (+ vert_diff.F90 for
!! surf_diff_type) against fms/mpp stubs and a stub of the model plumbing
!! (transforms/spectral_dynamics/mpp_domains/qflux/interpolator/diag) -- all
!! inert on the Frierson slab-ocean path. Tri_surf is allocated by vert_diff_init
!! and its fields set to plausible values; the surface fluxes are set directly.
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     -fallow-argument-mismatch \
!!     fms_stubs.F90 mpp_mod_stub.F90 mixed_layer_stubs.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_param/vert_diff/vert_diff.F90 \
!!     $ISCA_SRC/src/atmos_spectral/driver/solo/mixed_layer.F90 \
!!     dump_mixed_layer_reference.F90 -o dump_mixed_layer_reference
!! Run with JSCA_DUMP_DIR set, then convert with read_dumps.py.

program dump_mixed_layer_reference
  use     vert_diff_mod, only: vert_diff_init, surf_diff_type
  use   mixed_layer_mod, only: mixed_layer_init, mixed_layer
  use time_manager_mod, only: time_type
  use    jsca_dump_mod, only: jsca_dump_2d

  implicit none

  integer, parameter :: nlon = 4, nlat = 6, nz = 20
  integer :: axes(4) = (/1, 2, 3, 4/)
  type(time_type)      :: Time
  type(surf_diff_type) :: Tri_surf
  real, dimension(nlon,nlat)   :: t_surf, t_surf_in, flux_t, flux_q, flux_r, &
       net_sw, lw_down, dhdt_surf, dedt_surf, dedq_surf, drdt_surf, dhdt_atm, &
       dedq_atm, albedo, albedo_out
  real, dimension(nlon,nlat)   :: delta_t_in, delta_q_in
  real, dimension(nlon,nlat,1) :: bucket_depth
  real, dimension(nlon+1,nlat+1) :: rad_lonb, rad_latb
  logical, dimension(nlon,nlat)  :: land
  real    :: dt, frac
  integer :: i, j, unit

  Time = time_type(0, 0)
  dt = 720.0
  land = .false.
  rad_lonb = 0.0; rad_latb = 0.0
  albedo = 0.31

  ! --- Frierson mixed_layer namelist (real input.nml, read via open_namelist_file) ---
  open(newunit=unit, file='input.nml', status='replace')
  write(unit,'(a)') '&mixed_layer_nml'
  write(unit,'(a)') '  depth = 2.5, albedo_value = 0.31, evaporation = .true.,'
  write(unit,'(a)') '  tconst = 285.0, prescribe_initial_dist = .true.'
  write(unit,'(a)') '/'
  close(unit)

  ! allocate Tri_surf (vert_diff owns surf_diff_type)
  call vert_diff_init(Tri_surf, nlon, nlat, nz, .true., .false.)
  call mixed_layer_init(1, nlon, 1, nlat, nz, t_surf, bucket_depth, axes, Time, &
       albedo, rad_lonb, rad_latb, land, .false.)

  ! --- set the surface state, fluxes and Tri_surf coupling terms ---
  do j = 1, nlat
    do i = 1, nlon
      frac = real((i-1) + (j-1)*nlon) / real(nlon*nlat - 1)   ! 0..1 across grid
      t_surf(i,j)    = 290.0 + 8.0 * frac                     ! 290..298 K
      ! surface fluxes (as surface_flux would return)
      flux_t(i,j)    = -30.0 + 80.0 * frac                    ! sensible (W/m^2)
      flux_q(i,j)    =  2.0e-5 + 8.0e-5 * frac                ! evaporation (kg/m^2/s)
      flux_r(i,j)    = 5.6734e-8 * t_surf(i,j)**4             ! upward LW
      net_sw(i,j)    = 150.0 + 120.0 * frac                   ! net SW down
      lw_down(i,j)   = 300.0 + 60.0 * frac                    ! LW down
      dhdt_surf(i,j) = 4.0 + 3.0 * frac
      dedt_surf(i,j) = 5.0e-6 + 5.0e-6 * frac
      dedq_surf(i,j) = 0.0
      drdt_surf(i,j) = 4.0 * 5.6734e-8 * t_surf(i,j)**3
      dhdt_atm(i,j)  = -(4.0 + 3.0 * frac)
      dedq_atm(i,j)  = -0.04 - 0.02 * frac
      ! vert_diff surface coupling (plausible mu_delt_n / -nu(1-e) / stored deltas)
      Tri_surf%dtmass(i,j)      = 1.2 + 0.4 * frac
      Tri_surf%dflux_t(i,j)     = -0.010 - 0.004 * frac
      Tri_surf%dflux_tr(i,j,1)  = -0.010 - 0.004 * frac
      Tri_surf%delta_t(i,j)     = 0.02 * (frac - 0.5)
      Tri_surf%delta_tr(i,j,1)  = 3.0e-5 * (frac - 0.5)
    end do
  end do
  t_surf_in = t_surf
  delta_t_in = Tri_surf%delta_t
  delta_q_in = Tri_surf%delta_tr(:,:,1)

  call mixed_layer(Time, Time, 1, nlat, t_surf, flux_t, flux_q, flux_r, dt, &
       net_sw, lw_down, Tri_surf, dhdt_surf, dedt_surf, dedq_surf, drdt_surf, &
       dhdt_atm, dedq_atm, albedo_out)

  call jsca_dump_2d('ml_t_surf_in',  t_surf_in)
  call jsca_dump_2d('ml_flux_t',     flux_t)
  call jsca_dump_2d('ml_flux_q',     flux_q)
  call jsca_dump_2d('ml_flux_r',     flux_r)
  call jsca_dump_2d('ml_net_sw',     net_sw)
  call jsca_dump_2d('ml_lw_down',    lw_down)
  call jsca_dump_2d('ml_dhdt_surf',  dhdt_surf)
  call jsca_dump_2d('ml_dedt_surf',  dedt_surf)
  call jsca_dump_2d('ml_drdt_surf',  drdt_surf)
  call jsca_dump_2d('ml_dhdt_atm',   dhdt_atm)
  call jsca_dump_2d('ml_dedq_atm',   dedq_atm)
  call jsca_dump_2d('ml_dtmass',     Tri_surf%dtmass)
  call jsca_dump_2d('ml_dflux_t',    Tri_surf%dflux_t)
  call jsca_dump_2d('ml_dflux_q',    Tri_surf%dflux_tr(:,:,1))
  call jsca_dump_2d('ml_delta_t_in', delta_t_in)   ! Tri_surf%delta_* are inout
  call jsca_dump_2d('ml_delta_q_in', delta_q_in)
  call jsca_dump_2d('ml_t_surf_out', t_surf)
  call jsca_dump_2d('ml_delta_t_out', Tri_surf%delta_t)
  call jsca_dump_2d('ml_delta_q_out', Tri_surf%delta_tr(:,:,1))

  write(*,*) 'ML_DUMP_DONE  dTs range', minval(t_surf-t_surf_in), maxval(t_surf-t_surf_in)
end program dump_mixed_layer_reference
