!! Infrastructure stubs for compiling damping_driver.f90 standalone.
!!
!! For the Frierson aquaplanet only the Rayleigh sponge is active
!! (do_rayleigh=.true.; do_mg_drag/do_cg_drag/do_topo_drag/do_const_drag all
!! .false.), so the gravity-wave-drag modules and the diagnostic sends are never
!! executed -- register_diag_field returns -1, so every `if (id_* > 0)` send is
!! skipped. These stubs just supply the symbols so the real, unmodified
!! damping_driver.f90 compiles and links. No numerics.
!!
!! (fms_mod / constants_mod come from fms_stubs.F90.)

module time_manager_mod
implicit none
private
public :: time_type, get_time, length_of_year
type time_type
  integer :: seconds = 0, days = 0
end type time_type
contains
subroutine get_time(Time, seconds, days)
  type(time_type), intent(in) :: Time
  integer, intent(out) :: seconds
  integer, intent(out), optional :: days
  seconds = Time%seconds
  if (present(days)) days = Time%days
end subroutine get_time
function length_of_year() result(yr)
  type(time_type) :: yr
  yr = time_type(0, 365)     ! only used on the do_const_drag path (not exercised)
end function length_of_year
end module time_manager_mod

!-------------------------------------------------------------------------------
module diag_manager_mod
use time_manager_mod, only: time_type
implicit none
private
public :: register_diag_field, register_static_field, send_data
interface send_data
  module procedure send_2d, send_3d
end interface
contains
! Return -1 so the caller's `if (id_* > 0)` diagnostic sends are all skipped.
integer function register_diag_field(module_name, field_name, axes, init_time, &
                                     long_name, units, missing_value)
  character(len=*), intent(in) :: module_name, field_name
  integer, intent(in) :: axes(:)
  type(time_type), intent(in) :: init_time
  character(len=*), intent(in), optional :: long_name, units
  real, intent(in), optional :: missing_value
  register_diag_field = -1
end function register_diag_field
integer function register_static_field(module_name, field_name, axes, &
                                        long_name, units, missing_value)
  character(len=*), intent(in) :: module_name, field_name
  integer, intent(in) :: axes(:)
  character(len=*), intent(in), optional :: long_name, units
  real, intent(in), optional :: missing_value
  register_static_field = -1
end function register_static_field
logical function send_2d(id, field, time, rmask)
  integer, intent(in) :: id
  real, intent(in) :: field(:,:)
  type(time_type), intent(in), optional :: time
  real, intent(in), optional :: rmask(:,:)
  send_2d = .true.
end function send_2d
logical function send_3d(id, field, time, rmask)
  integer, intent(in) :: id
  real, intent(in) :: field(:,:,:)
  type(time_type), intent(in), optional :: time
  real, intent(in), optional :: rmask(:,:,:)
  send_3d = .true.
end function send_3d
end module diag_manager_mod

!-------------------------------------------------------------------------------
! Gravity-wave-drag modules: link-only (all three flags .false. for Frierson).
module mg_drag_mod
implicit none
private
public :: mg_drag, mg_drag_init, mg_drag_end
contains
subroutine mg_drag_init(lonb, latb, sgsmtn)
  real, intent(in) :: lonb(:), latb(:)
  real, intent(out) :: sgsmtn(:,:)
  sgsmtn = 0.0
end subroutine mg_drag_init
subroutine mg_drag(is, js, delt, u, v, t, pfull, phalf, zfull, zhalf, &
                   utnd, vtnd, ttnd, taubx, tauby, taus, kbot)
  integer, intent(in) :: is, js
  real, intent(in) :: delt
  real, intent(in), dimension(:,:,:) :: u, v, t, pfull, phalf, zfull, zhalf
  real, intent(out), dimension(:,:,:) :: utnd, vtnd, ttnd, taus
  real, intent(out), dimension(:,:) :: taubx, tauby
  integer, intent(in), optional :: kbot(:,:)
  utnd = 0.0; vtnd = 0.0; ttnd = 0.0; taus = 0.0; taubx = 0.0; tauby = 0.0
end subroutine mg_drag
subroutine mg_drag_end()
end subroutine mg_drag_end
end module mg_drag_mod

!-------------------------------------------------------------------------------
module cg_drag_mod
use time_manager_mod, only: time_type
implicit none
private
public :: cg_drag_init, cg_drag_calc, cg_drag_end
contains
subroutine cg_drag_init(lonb, latb, pref, Time, axes)
  real, intent(in) :: lonb(:), latb(:), pref(:)
  type(time_type), intent(in), optional :: Time
  integer, intent(in), optional :: axes(:)
end subroutine cg_drag_init
subroutine cg_drag_calc(is, js, lat, pfull, zfull, t, u, v, Time, delt, utnd, vtnd)
  integer, intent(in) :: is, js
  real, intent(in) :: lat(:,:)
  real, intent(in), dimension(:,:,:) :: pfull, zfull, t, u, v
  type(time_type), intent(in) :: Time
  real, intent(in) :: delt
  real, intent(out), dimension(:,:,:) :: utnd, vtnd
  utnd = 0.0; vtnd = 0.0
end subroutine cg_drag_calc
subroutine cg_drag_end()
end subroutine cg_drag_end
end module cg_drag_mod

!-------------------------------------------------------------------------------
module topo_drag_mod
implicit none
private
public :: topo_drag_init, topo_drag, topo_drag_end
contains
subroutine topo_drag_init(lonb, latb, ierr)
  real, intent(in) :: lonb(:), latb(:)
  integer, intent(out) :: ierr
  ierr = 0
end subroutine topo_drag_init
subroutine topo_drag(is, js, u, v, t, pfull, phalf, zfull, zhalf, &
                     z_pbl, taubx, tauby, utnd, vtnd, taus)
  integer, intent(in) :: is, js
  real, intent(in), dimension(:,:,:) :: u, v, t, pfull, phalf, zfull, zhalf
  real, intent(in), dimension(:,:) :: z_pbl
  real, intent(out), dimension(:,:) :: taubx, tauby
  real, intent(out), dimension(:,:,:) :: utnd, vtnd, taus
  utnd = 0.0; vtnd = 0.0; taus = 0.0; taubx = 0.0; tauby = 0.0
end subroutine topo_drag
subroutine topo_drag_end()
end subroutine topo_drag_end
end module topo_drag_mod
