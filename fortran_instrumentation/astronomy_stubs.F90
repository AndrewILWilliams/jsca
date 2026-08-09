!! Stubs for the dependencies of the REAL, unmodified astronomy.f90 that are not
!! provided by fms_stubs.F90 (which supplies fms_mod + constants_mod). Only the
!! symbols astronomy's `use, only:` lists require are defined; the diurnal_solar
!! path exercised by the fixture driver never calls the time_manager routines
!! (it is handed gmt / time_since_ae directly), so those are no-op / trivial.
!!
!! time_manager_mod: astronomy imports a broad set (set_time, get_date_julian,
!! set_date_julian, set_date, length_of_year/day, time_manager_init, the +,-,//,<
!! operators, get_calendar_type, NO_CALENDAR). Only the type needs to be real;
!! the rest are stubs to satisfy the `use` at compile time.
!!
!! mpp_mod: only input_nml_file (a namelist string buffer) is imported.

module mpp_mod
implicit none
character(len=1), dimension(1), public :: input_nml_file = (/' '/)
end module mpp_mod


module time_manager_mod
implicit none
private
public :: time_type, set_time, get_time, get_date_julian, set_date_julian, &
          set_date, length_of_year, length_of_day, time_manager_init, &
          get_calendar_type, NO_CALENDAR
public :: operator(+), operator(-), operator(//), operator(<)

integer, parameter :: NO_CALENDAR = 0

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
interface operator(//)
  module procedure t_concat
end interface
interface operator(<)
  module procedure t_lt
end interface

contains

type(time_type) function set_time(seconds, days)
  integer, intent(in) :: seconds
  integer, intent(in), optional :: days
  set_time%seconds = seconds
  set_time%days = 0
  if (present(days)) set_time%days = days
end function set_time

subroutine get_time_3(t, seconds, days)
  type(time_type), intent(in) :: t
  integer, intent(out) :: seconds, days
  seconds = t%seconds; days = t%days
end subroutine get_time_3

subroutine get_time_2(t, seconds)
  type(time_type), intent(in) :: t
  integer, intent(out) :: seconds
  seconds = t%seconds
end subroutine get_time_2

subroutine get_date_julian(t, yr, mo, dy, hr, mn, sc)
  type(time_type), intent(in) :: t
  integer, intent(out) :: yr, mo, dy, hr, mn, sc
  yr = 0; mo = 0; dy = 0; hr = 0; mn = 0; sc = 0
end subroutine get_date_julian

type(time_type) function set_date_julian(yr, mo, dy, hr, mn, sc)
  integer, intent(in) :: yr, mo, dy
  integer, intent(in), optional :: hr, mn, sc
  set_date_julian = time_type(0, 0)
end function set_date_julian

type(time_type) function set_date(yr, mo, dy, hr, mn, sc)
  integer, intent(in) :: yr, mo, dy
  integer, intent(in), optional :: hr, mn, sc
  set_date = time_type(0, 0)
end function set_date

type(time_type) function length_of_year()
  length_of_year = time_type(0, 365)
end function length_of_year

real function length_of_day()
  length_of_day = 86400.0
end function length_of_day

subroutine time_manager_init
end subroutine time_manager_init

integer function get_calendar_type()
  get_calendar_type = NO_CALENDAR
end function get_calendar_type

type(time_type) function t_add(a, b)
  type(time_type), intent(in) :: a, b
  t_add = time_type(a%seconds + b%seconds, a%days + b%days)
end function t_add

type(time_type) function t_sub(a, b)
  type(time_type), intent(in) :: a, b
  t_sub = time_type(a%seconds - b%seconds, a%days - b%days)
end function t_sub

! FMS overloads operator(//) on time_type as time division: a // b = a/b as a
! real ratio (used by astronomy's calendar/orbital_time paths, not the diurnal
! path exercised here). Return the ratio of total seconds.
real function t_concat(a, b)
  type(time_type), intent(in) :: a, b
  real :: bs
  bs = real(b%days) * 86400.0 + real(b%seconds)
  if (bs == 0.0) bs = 1.0
  t_concat = (real(a%days) * 86400.0 + real(a%seconds)) / bs
end function t_concat

logical function t_lt(a, b)
  type(time_type), intent(in) :: a, b
  t_lt = (a%days < b%days) .or. (a%days == b%days .and. a%seconds < b%seconds)
end function t_lt

end module time_manager_mod
