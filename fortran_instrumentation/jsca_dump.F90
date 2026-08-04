!! jsca_dump_mod — Tier-1 golden-fixture harvester for the Isca Fortran baseline.
!!
!! Drop this file into the Isca source tree (e.g. src/shared/jsca_dump/) and add
!! it to path_names, then call jsca_dump_* around any routine you want fixtures
!! for. Dumps are raw little-endian float64 stream files plus a one-line text
!! manifest each; read them in Python with fortran_instrumentation/read_dumps.py.
!!
!! Activation: set the environment variable JSCA_DUMP_DIR to a writable
!! directory. Unset => every call is a no-op, so instrumented builds can run
!! production cases unchanged.
!!
!! Deliberately dependency-free (no netCDF, no FMS) so it can be added to any
!! Isca build — including per-PE builds, where each rank writes its own files
!! (suffix _peNNN); harvest single-PE runs to keep fixtures global.
!!
!! Isca compiles with -fdefault-real-8, so `real` below is 8-byte, matching
!! numpy float64 exactly.

module jsca_dump_mod
implicit none
private
public :: jsca_dump_1d, jsca_dump_2d, jsca_dump_3d, jsca_dump_4d, jsca_dump_scalar

integer, save :: counter = 0
logical, save :: initialized = .false.
logical, save :: active = .false.
character(len=512), save :: dump_dir = ''
integer, save :: pe = 0

contains

subroutine init_once(mype)
  integer, intent(in), optional :: mype
  integer :: stat
  if (initialized) return
  call get_environment_variable('JSCA_DUMP_DIR', dump_dir, status=stat)
  active = (stat == 0 .and. len_trim(dump_dir) > 0)
  if (present(mype)) pe = mype
  initialized = .true.
end subroutine init_once

subroutine write_manifest(name, dims, fname)
  character(len=*), intent(in) :: name, fname
  integer, intent(in) :: dims(:)
  integer :: mu
  open(newunit=mu, file=trim(fname)//'.meta', form='formatted', status='replace')
  write(mu, '(A)') trim(name)
  write(mu, '(A)') 'float64'
  write(mu, *) size(dims), dims
  close(mu)
end subroutine write_manifest

function next_fname(name) result(fname)
  character(len=*), intent(in) :: name
  character(len=600) :: fname
  counter = counter + 1
  write(fname, '(A,"/jsca_",I6.6,"_",A,"_pe",I3.3)') trim(dump_dir), counter, trim(name), pe
end function next_fname

subroutine jsca_dump_scalar(name, x)
  character(len=*), intent(in) :: name
  real, intent(in) :: x
  real :: buf(1)
  buf(1) = x
  call jsca_dump_1d(name, buf)
end subroutine jsca_dump_scalar

subroutine jsca_dump_1d(name, x)
  character(len=*), intent(in) :: name
  real, intent(in) :: x(:)
  character(len=600) :: fname
  integer :: u
  call init_once()
  if (.not. active) return
  fname = next_fname(name)
  call write_manifest(name, shape(x), fname)
  open(newunit=u, file=trim(fname)//'.bin', form='unformatted', access='stream', status='replace')
  write(u) x
  close(u)
end subroutine jsca_dump_1d

subroutine jsca_dump_2d(name, x)
  character(len=*), intent(in) :: name
  real, intent(in) :: x(:,:)
  character(len=600) :: fname
  integer :: u
  call init_once()
  if (.not. active) return
  fname = next_fname(name)
  call write_manifest(name, shape(x), fname)
  open(newunit=u, file=trim(fname)//'.bin', form='unformatted', access='stream', status='replace')
  write(u) x
  close(u)
end subroutine jsca_dump_2d

subroutine jsca_dump_3d(name, x)
  character(len=*), intent(in) :: name
  real, intent(in) :: x(:,:,:)
  character(len=600) :: fname
  integer :: u
  call init_once()
  if (.not. active) return
  fname = next_fname(name)
  call write_manifest(name, shape(x), fname)
  open(newunit=u, file=trim(fname)//'.bin', form='unformatted', access='stream', status='replace')
  write(u) x
  close(u)
end subroutine jsca_dump_3d

subroutine jsca_dump_4d(name, x)
  character(len=*), intent(in) :: name
  real, intent(in) :: x(:,:,:,:)
  character(len=600) :: fname
  integer :: u
  call init_once()
  if (.not. active) return
  fname = next_fname(name)
  call write_manifest(name, shape(x), fname)
  open(newunit=u, file=trim(fname)//'.bin', form='unformatted', access='stream', status='replace')
  write(u) x
  close(u)
end subroutine jsca_dump_4d

end module jsca_dump_mod
