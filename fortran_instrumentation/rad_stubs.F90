!! Infrastructure stubs for compiling two_stream_gray_rad.F90 standalone.
!!
!! two_stream_gray_rad pulls in diag_manager / time_manager / astronomy /
!! interpolator, but for the Frierson scheme (rad_scheme='frierson',
!! do_seasonal=.false., do_read_co2=.false.) none of their NUMERICS are on the
!! computed path: diagnostics are disabled (register_diag_field -> -1 guards all
!! send_data off), astronomy/time_manager are only used in the do_seasonal
!! branch, and the interpolator only if do_read_co2. These supply the symbols so
!! the real module compiles and runs unmodified; no reimplemented numerics.
!!
!! (fms_mod / constants_mod come from fms_stubs.F90.)

module time_manager_mod
implicit none
private
public :: time_type, get_time, length_of_year, length_of_day
public :: operator(+), operator(-), operator(/=)

type time_type
  integer :: seconds = 0, days = 0
end type time_type

interface get_time
  module procedure get_time_3, get_time_2
end interface
interface operator(+)
  module procedure t_add
end interface
interface operator(-)
  module procedure t_sub
end interface
interface operator(/=)
  module procedure t_ne
end interface

contains
subroutine get_time_3(t, seconds, days)
  type(time_type), intent(in) :: t
  integer, intent(out) :: seconds
  integer, intent(out) :: days
  seconds = t%seconds; days = t%days
end subroutine get_time_3
subroutine get_time_2(t, seconds)
  type(time_type), intent(in) :: t
  integer, intent(out) :: seconds
  seconds = t%seconds
end subroutine get_time_2
type(time_type) function length_of_year()
  length_of_year = time_type(0, 365)
end function length_of_year
real function length_of_day()
  length_of_day = 86400.0
end function length_of_day
type(time_type) function t_add(a, b)
  type(time_type), intent(in) :: a, b
  t_add = time_type(a%seconds + b%seconds, a%days + b%days)
end function t_add
type(time_type) function t_sub(a, b)
  type(time_type), intent(in) :: a, b
  t_sub = time_type(a%seconds - b%seconds, a%days - b%days)
end function t_sub
logical function t_ne(a, b)
  type(time_type), intent(in) :: a, b
  t_ne = (a%seconds /= b%seconds) .or. (a%days /= b%days)
end function t_ne
end module time_manager_mod

!-------------------------------------------------------------------------------
module diag_manager_mod
use time_manager_mod, only: time_type
implicit none
private
public :: register_diag_field, send_data

interface register_diag_field
  module procedure reg_ax, reg_noax
end interface
interface send_data
  module procedure send_0d, send_2d, send_3d
end interface

contains
integer function reg_ax(module_name, field_name, axes, init_time, &
                        long_name, units, missing_value)
  character(len=*), intent(in) :: module_name, field_name
  integer, intent(in) :: axes(:)
  type(time_type), intent(in) :: init_time
  character(len=*), intent(in), optional :: long_name, units
  real, intent(in), optional :: missing_value
  reg_ax = -1   ! disabled -> all send_data calls guarded off
end function reg_ax
integer function reg_noax(module_name, field_name, init_time, &
                          long_name, units, missing_value)
  character(len=*), intent(in) :: module_name, field_name
  type(time_type), intent(in) :: init_time
  character(len=*), intent(in), optional :: long_name, units
  real, intent(in), optional :: missing_value
  reg_noax = -1
end function reg_noax
logical function send_0d(id, field, time)
  integer, intent(in) :: id
  real, intent(in) :: field
  type(time_type), intent(in), optional :: time
  send_0d = .true.
end function send_0d
logical function send_2d(id, field, time)
  integer, intent(in) :: id
  real, intent(in) :: field(:,:)
  type(time_type), intent(in), optional :: time
  send_2d = .true.
end function send_2d
logical function send_3d(id, field, time)
  integer, intent(in) :: id
  real, intent(in) :: field(:,:,:)
  type(time_type), intent(in), optional :: time
  send_3d = .true.
end function send_3d
end module diag_manager_mod

!-------------------------------------------------------------------------------
module astronomy_mod
use time_manager_mod, only: time_type
implicit none
private
public :: astronomy_init, diurnal_solar

interface diurnal_solar
  module procedure diurnal_solar_2d
end interface

contains
subroutine astronomy_init()
end subroutine astronomy_init
! Only invoked on the do_seasonal path (not exercised by the Frierson fixture).
subroutine diurnal_solar_2d(lat, lon, gmt, time_since_ae, coszen, fracsun, rrsun, dt)
  real, intent(in) :: lat(:,:), lon(:,:)
  real, intent(in) :: gmt, time_since_ae
  real, intent(out) :: coszen(:,:), fracsun(:,:)
  real, intent(out) :: rrsun
  real, intent(in), optional :: dt
  coszen = 0.0; fracsun = 0.0; rrsun = 1.0
end subroutine diurnal_solar_2d
end module astronomy_mod

!-------------------------------------------------------------------------------
module interpolator_mod
use time_manager_mod, only: time_type
implicit none
private
public :: interpolate_type, interpolator_init, interpolator, interpolator_end, ZERO

integer, parameter :: ZERO = 0   ! out-of-bounds flag (do_read_co2 path only)

type interpolate_type
  integer :: dummy = 0
end type interpolate_type

interface interpolator
  module procedure interpolator_3d
end interface

contains
subroutine interpolator_init(clim_type, file_name, lonb, latb, data_out_of_bounds)
  type(interpolate_type), intent(inout) :: clim_type
  character(len=*), intent(in) :: file_name
  real, intent(in), dimension(:,:) :: lonb, latb
  integer, intent(in), dimension(:) :: data_out_of_bounds
end subroutine interpolator_init
subroutine interpolator_3d(clim_type, Time, phalf, interp_data, field_name)
  type(interpolate_type), intent(inout) :: clim_type
  type(time_type), intent(in) :: Time
  real, intent(in), dimension(:,:,:) :: phalf
  real, intent(out), dimension(:,:,:) :: interp_data
  character(len=*), intent(in) :: field_name
  interp_data = 0.0
end subroutine interpolator_3d
subroutine interpolator_end(clim_type)
  type(interpolate_type), intent(inout) :: clim_type
end subroutine interpolator_end
end module interpolator_mod
