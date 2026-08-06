!! Golden-fixture driver for Isca's bulk surface fluxes (surface_flux.F90), the
!! Frierson aquaplanet ocean path: surface_flux_nml do_simple=.true.,
!! use_virtual_temp=.false., old_dtaudv=.true.; all points ocean (land=F,
!! seawater=T, avail=T), no bucket.
!!
!! Compiles the REAL, unmodified surface_flux.F90 together with the real
!! monin_obukhov wrapper (+ its _PURE kernel) and the real sat_vapor_pres wrapper
!! (+ kernel), against the shared fms/mpp/mpp_io stubs. Namelists are injected
!! through FMS's internal-file buffer (build -DINTERNAL_FILE_NML).
!!
!! Inputs span a range of air-sea temperature/humidity differences and winds so
!! the sensible/latent/momentum fluxes and their implicit derivatives are all
!! exercised. Dumps the model-facing flux outputs + derivatives for
!! tests/test_surface_flux_fixtures.py.
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none -cpp \
!!     -DINTERNAL_FILE_NML \
!!     -I$ISCA_SRC/src/shared/include -I$ISCA_SRC/src/atmos_param/monin_obukhov \
!!     fms_stubs.F90 mpp_mod_stub.F90 mpp_io_stub.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_param/monin_obukhov/monin_obukhov_kernel.F90 \
!!     $ISCA_SRC/src/atmos_param/monin_obukhov/monin_obukhov.F90 \
!!     $ISCA_SRC/src/shared/sat_vapor_pres/sat_vapor_pres_k.F90 \
!!     $ISCA_SRC/src/shared/sat_vapor_pres/sat_vapor_pres.F90 \
!!     $ISCA_SRC/src/coupler/surface_flux.F90 \
!!     dump_surface_flux_reference.F90 -o dump_surface_flux_reference
!! Run with JSCA_DUMP_DIR set, then convert with read_dumps.py.

program dump_surface_flux_reference
  use            mpp_mod, only: input_nml_file
  use   surface_flux_mod, only: surface_flux
  use      jsca_dump_mod, only: jsca_dump_1d
  implicit none

  integer, parameter :: n = 40
  real, dimension(n) :: t_atm, q_atm, u_atm, v_atm, p_atm, z_atm, p_surf, t_surf, &
       t_ca, q_surf, u_surf, v_surf, rough_mom, rough_heat, rough_moist, &
       rough_scale, gust
  real, dimension(n) :: flux_t, flux_q, flux_r, flux_u, flux_v, cd_m, cd_t, cd_q, &
       w_atm, u_star, b_star, q_star, dhdt_surf, dedt_surf, dedq_surf, drdt_surf, &
       dhdt_atm, dedq_atm, dtaudu_atm, dtaudv_atm, ex_del_m, ex_del_h, ex_del_q, &
       temp_2m, u_10m, v_10m, q_2m, rh_2m
  real, dimension(n) :: bucket_depth, depth_change_lh, depth_change_conv, depth_change_cond
  logical, dimension(n) :: land, seawater, avail
  real    :: dt, max_bucket_depth_land, dtheta, spd, eps, ess
  integer :: i, unit

  ! --- inject the Frierson surface_flux + sat_vapor_pres namelists ---
  allocate(input_nml_file(3))
  input_nml_file(1) = '&surface_flux_nml do_simple=.true. use_virtual_temp=.false. old_dtaudv=.true. /'
  input_nml_file(2) = '&sat_vapor_pres_nml do_simple=.true. /'
  input_nml_file(3) = ' '

  eps = 287.04 / 461.50
  dt = 720.0
  land = .false.; seawater = .true.; avail = .true.
  max_bucket_depth_land = 2.0
  bucket_depth = 0.0; depth_change_lh = 0.0; depth_change_conv = 0.0; depth_change_cond = 0.0
  u_surf = 0.0; v_surf = 0.0
  rough_mom = 3.21e-5; rough_heat = 3.21e-5; rough_moist = 3.21e-5
  rough_scale = 3.21e-5   ! driver passes rough_mom as rough_scale -> ratio 1
  gust = 1.0

  do i = 1, n
    ! air-sea contrast sweeps unstable..stable; wind sweeps 1..16 m/s
    dtheta = -4.0 + 12.0 * real(mod(i-1, 10)) / 9.0        ! t_atm - t_surf: -4..+8 K
    spd    = 1.0 + 15.0 * real((i-1) / 10) / 3.0            ! 1..16 m/s (4 bands)
    t_surf(i) = 298.0 - 8.0 * real((i-1)/10) / 3.0          ! 298..290 K
    t_atm(i)  = t_surf(i) + dtheta
    t_ca(i)   = t_surf(i)
    u_atm(i)  = spd
    v_atm(i)  = 0.0
    p_surf(i) = 1.0e5
    p_atm(i)  = 0.99e5
    z_atm(i)  = 30.0
    ! atmospheric humidity ~ 70% of saturation at the air temperature (do_simple es)
    ess = 610.78 * exp(-2.5e6/461.50 * (1.0/t_atm(i) - 1.0/273.16))
    q_atm(i)  = 0.70 * eps * ess / p_atm(i)
    q_surf(i) = q_atm(i)   ! input q_surf (used only in the q_2m diagnostic)
  end do

  call surface_flux(t_atm, q_atm, u_atm, v_atm, p_atm, z_atm, p_surf, t_surf, &
       t_ca, q_surf, .false., bucket_depth, max_bucket_depth_land, &
       depth_change_lh, depth_change_conv, depth_change_cond, u_surf, v_surf, &
       rough_mom, rough_heat, rough_moist, rough_scale, gust, &
       flux_t, flux_q, flux_r, flux_u, flux_v, cd_m, cd_t, cd_q, &
       w_atm, u_star, b_star, q_star, dhdt_surf, dedt_surf, dedq_surf, drdt_surf, &
       dhdt_atm, dedq_atm, dtaudu_atm, dtaudv_atm, ex_del_m, ex_del_h, ex_del_q, &
       temp_2m, u_10m, v_10m, q_2m, rh_2m, dt, land, seawater, avail)

  call jsca_dump_1d('sf_t_atm',  t_atm)
  call jsca_dump_1d('sf_q_atm',  q_atm)
  call jsca_dump_1d('sf_u_atm',  u_atm)
  call jsca_dump_1d('sf_v_atm',  v_atm)
  call jsca_dump_1d('sf_p_atm',  p_atm)
  call jsca_dump_1d('sf_z_atm',  z_atm)
  call jsca_dump_1d('sf_p_surf', p_surf)
  call jsca_dump_1d('sf_t_surf', t_surf)
  call jsca_dump_1d('sf_q_surf_in', q_atm)   ! q_surf input equals q_atm here
  call jsca_dump_1d('sf_rough_mom',   rough_mom)
  call jsca_dump_1d('sf_rough_heat',  rough_heat)
  call jsca_dump_1d('sf_rough_moist', rough_moist)
  call jsca_dump_1d('sf_gust',   gust)
  call jsca_dump_1d('sf_flux_t', flux_t)
  call jsca_dump_1d('sf_flux_q', flux_q)
  call jsca_dump_1d('sf_flux_r', flux_r)
  call jsca_dump_1d('sf_flux_u', flux_u)
  call jsca_dump_1d('sf_flux_v', flux_v)
  call jsca_dump_1d('sf_cd_m',   cd_m)
  call jsca_dump_1d('sf_cd_t',   cd_t)
  call jsca_dump_1d('sf_cd_q',   cd_q)
  call jsca_dump_1d('sf_w_atm',  w_atm)
  call jsca_dump_1d('sf_u_star', u_star)
  call jsca_dump_1d('sf_b_star', b_star)
  call jsca_dump_1d('sf_q_star', q_star)
  call jsca_dump_1d('sf_dhdt_surf', dhdt_surf)
  call jsca_dump_1d('sf_dedt_surf', dedt_surf)
  call jsca_dump_1d('sf_dedq_atm',  dedq_atm)
  call jsca_dump_1d('sf_drdt_surf', drdt_surf)
  call jsca_dump_1d('sf_dhdt_atm',  dhdt_atm)
  call jsca_dump_1d('sf_dtaudu_atm', dtaudu_atm)
  call jsca_dump_1d('sf_temp_2m', temp_2m)
  call jsca_dump_1d('sf_u_10m',   u_10m)
  call jsca_dump_1d('sf_q_2m',    q_2m)
  call jsca_dump_1d('sf_rh_2m',   rh_2m)

  write(*,*) 'SF_DUMP_DONE flux_t min/max:', minval(flux_t), maxval(flux_t)
  write(*,*) '  flux_q min/max:', minval(flux_q), maxval(flux_q)
end program dump_surface_flux_reference
