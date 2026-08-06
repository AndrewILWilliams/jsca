!! Golden-fixture driver for Isca's Monin-Obukhov surface-layer kernel
!! (monin_obukhov_kernel.F90), with the default monin_obukhov_nml the Frierson
!! aquaplanet uses (stable_option=1, rich_crit=2.0, zeta_trans=0.5, neutral=F,
!! drag_min=1e-5; solver error=1e-4, zeta_min=1e-6, small=1e-4, max_iter=20).
!!
!! The kernel routines are _PURE and dependency-free, so this calls them directly
!! (via the monin_obukhov_inter interface module) -- no stubs. Inputs span the
!! unstable / neutral / stable / near-critical-stable regimes so every branch of
!! the stability functions and the Newton zeta-solver is exercised. Dumps the
!! drag coefficients + friction scales (mo_drag) and the reference-height ratios
!! (mo_profile) for tests/test_monin_obukhov_fixtures.py.
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none -cpp \
!!     -I$ISCA_SRC/src/shared/include \
!!     -I$ISCA_SRC/src/atmos_param/monin_obukhov \
!!     $ISCA_SRC/src/atmos_param/monin_obukhov/monin_obukhov_kernel.F90 \
!!     jsca_dump.F90 dump_monin_obukhov_reference.F90 -o dump_monin_obukhov_reference
!! Run with JSCA_DUMP_DIR set, then convert with read_dumps.py.

program dump_monin_obukhov_reference
  use monin_obukhov_inter, only: monin_obukhov_drag_1d, monin_obukhov_profile_1d
  use          jsca_dump_mod, only: jsca_dump_1d
  implicit none

  ! monin_obukhov_nml defaults (Frierson uses these unmodified)
  real,    parameter :: grav = 9.80, vonkarm = 0.40
  real,    parameter :: error = 1.e-04, zeta_min = 1.e-06, small = 1.e-04
  integer, parameter :: max_iter = 20, stable_option = 1
  logical, parameter :: neutral = .false.
  real,    parameter :: rich_crit = 2.0, zeta_trans = 0.5, drag_min = 1.e-05
  real,    parameter :: zref = 10.0, zref_t = 2.0

  integer, parameter :: n = 60
  real, dimension(n) :: pt, pt0, z, z0, zt, zq, speed
  real, dimension(n) :: drag_m, drag_t, drag_q, u_star, b_star, q_star
  real, dimension(n) :: del_m, del_t, del_q
  logical, dimension(n) :: avail
  real    :: dtheta, spd
  integer :: i, ier

  avail = .true.

  ! Build a spread of surface-layer states. Index maps to (stability, wind):
  ! dtheta = pt0 - pt sweeps -8..+8 K (unstable..stable), speed sweeps 0.5..20.
  do i = 1, n
    dtheta = -8.0 + 16.0 * real(mod(i-1, 12)) / 11.0        ! -8 .. +8 K
    spd    = 0.5 + 19.5 * real((i-1) / 12) / 4.0             ! 0.5 .. 20 m/s (5 bands)
    pt(i)    = 285.0
    pt0(i)   = pt(i) + dtheta
    speed(i) = spd
    z(i)  = 30.0            ! lowest-level height (m)
    z0(i) = 3.21e-5        ! momentum roughness (ocean-like)
    zt(i) = 3.21e-5 * 0.5  ! heat roughness (distinct, to exercise f_t)
    zq(i) = 3.21e-5 * 2.0  ! moisture roughness (distinct, to exercise f_q)
  end do

  call monin_obukhov_drag_1d(grav, vonkarm, error, zeta_min, max_iter, small, &
       neutral, stable_option, rich_crit, zeta_trans, drag_min, &
       n, pt, pt0, z, z0, zt, zq, speed, drag_m, drag_t, drag_q, &
       u_star, b_star, .false., avail, ier)
  write(*,*) 'mo_drag ier =', ier

  q_star = 0.0   ! q_star is an argument but unused in the del computation
  call monin_obukhov_profile_1d(vonkarm, neutral, stable_option, rich_crit, &
       zeta_trans, n, zref, zref_t, z, z0, zt, zq, u_star, b_star, q_star, &
       del_m, del_t, del_q, .false., avail, ier)
  write(*,*) 'mo_profile ier =', ier

  call jsca_dump_1d('mo_pt',     pt)
  call jsca_dump_1d('mo_pt0',    pt0)
  call jsca_dump_1d('mo_z',      z)
  call jsca_dump_1d('mo_z0',     z0)
  call jsca_dump_1d('mo_zt',     zt)
  call jsca_dump_1d('mo_zq',     zq)
  call jsca_dump_1d('mo_speed',  speed)
  call jsca_dump_1d('mo_drag_m', drag_m)
  call jsca_dump_1d('mo_drag_t', drag_t)
  call jsca_dump_1d('mo_drag_q', drag_q)
  call jsca_dump_1d('mo_u_star', u_star)
  call jsca_dump_1d('mo_b_star', b_star)
  call jsca_dump_1d('mo_del_m',  del_m)
  call jsca_dump_1d('mo_del_t',  del_t)
  call jsca_dump_1d('mo_del_q',  del_q)
  write(*,*) 'MO_DUMP_DONE  drag_m min/max:', minval(drag_m), maxval(drag_m)
end program dump_monin_obukhov_reference
