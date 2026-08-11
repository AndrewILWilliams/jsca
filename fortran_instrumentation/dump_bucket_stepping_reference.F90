!! Golden-fixture driver for the bucket water-budget time-stepping of Isca's
!! idealized_moist_phys.F90 (the Manabe soil-moisture reservoir). The stepping is
!! the grid-space leapfrog + RAW filter applied to bucket_depth, followed by the
!! non-negativity clamp and the over-land runoff (capacity) clamp.
!!
!! The numeric body is the VERBATIM slice idealized_moist_phys.F90 L1401-1428,
!! extracted byte-for-byte via sed into bucket_stepping_body.inc and dropped into
!! the internal subroutine below, which supplies exactly the variable environment
!! the block would otherwise inherit from the (enormous, un-stubbable)
!! idealized_moist_phys module. NO numerics are reimplemented here.
!!
!! Two firings are dumped, matching the two branches of the leapfrog:
!!   * "step"  — the normal leapfrog step (previous /= current, future=previous),
!!               i.e. the value used on every timestep after the first;
!!   * "start" — the first Euler-ish step (previous == current, future=3-current).
!! Inputs span empty buckets (drive the <=0 clamp), below-capacity buckets (the
!! plain leapfrog range) and over-capacity buckets over both land (runoff clamp
!! fires) and ocean (runoff clamp must NOT fire).
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   export ISCA_SRC=/tmp/isca
!!   sed -n '1401,1428p' \
!!     $ISCA_SRC/src/atmos_spectral/driver/solo/idealized_moist_phys.F90 \
!!     > bucket_stepping_body.inc
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     jsca_dump.F90 dump_bucket_stepping_reference.F90 \
!!     -o dump_bucket_stepping_reference
!! Run with JSCA_DUMP_DIR set, then convert with read_dumps.py.

program dump_bucket_stepping_reference
  use jsca_dump_mod, only: jsca_dump_1d
  implicit none

  integer, parameter :: nx = 8, ny = 5, n = nx*ny
  ! time-level storage exactly as in idealized_moist_phys (num_time_levels = 2)
  real    :: bucket_depth(nx, ny, 2)
  real    :: dt_bucket(nx, ny), filt(nx, ny)
  real    :: depth_change_cond(nx, ny), depth_change_conv(nx, ny), depth_change_lh(nx, ny)
  logical :: land(nx, ny)
  real    :: robert_bucket, raw_bucket, max_bucket_depth_land
  integer :: previous, current, future
  ! saved inputs (2D) for the fixture
  real    :: prev_in(nx, ny), curr_in(nx, ny)
  integer :: i, j, idx

  robert_bucket = 0.04       ! idealized_moist_phys_nml default
  raw_bucket    = 0.53       ! idealized_moist_phys_nml default
  max_bucket_depth_land = 1.0

  ! --- input fields (shared by both firings) -------------------------------
  do j = 1, ny
    do i = 1, nx
      idx = (j-1)*nx + i                     ! 1..40
      ! previous-level depth 0 .. 1.6 (spans empty, below-cap, over-cap)
      prev_in(i, j) = 1.6 * real(idx-1) / real(n-1)
      ! current-level depth: a small offset from previous, still physical
      curr_in(i, j) = prev_in(i, j) + 0.05 * sin(real(idx))
      if (curr_in(i, j) < 0.0) curr_in(i, j) = 0.0
      ! precip inputs (positive) and evaporation loss (positive => drying)
      depth_change_cond(i, j) = 0.010 * real(mod(idx, 3))
      depth_change_conv(i, j) = 0.005 * real(mod(idx, 4))
      ! large LH loss on a subset drives net dt_bucket negative -> <=0 clamp
      depth_change_lh(i, j)   = 0.02 + 0.60 * real(mod(idx, 5)) / 4.0
      ! alternate land / ocean so the runoff clamp is exercised on land and
      ! deliberately skipped on ocean points that are over capacity
      land(i, j) = (mod(idx, 2) == 0)
    end do
  end do

  ! === firing 1: normal leapfrog step (previous /= current) ================
  previous = 1; current = 2
  future   = previous                        ! else-branch: future = previous
  bucket_depth(:, :, previous) = prev_in
  bucket_depth(:, :, current)  = curr_in
  call bucket_step_body()
  ! future slot (=previous index) holds the new bucket depth; current slot the
  ! RAW-filtered current-level value.
  call dump2d('bk_step_prev_in',  prev_in)
  call dump2d('bk_step_curr_in',  curr_in)
  call dump2d('bk_step_future',   bucket_depth(:, :, future))
  call dump2d('bk_step_current',  bucket_depth(:, :, current))

  ! === firing 2: first Euler-ish step (previous == current) ===============
  previous = 1; current = 1
  future   = 2                               ! if-branch: future = num_time_levels+1-current
  bucket_depth(:, :, 1) = prev_in            ! previous == current -> same slot
  bucket_depth(:, :, 2) = 0.0                ! scratch (overwritten as future)
  call bucket_step_body()
  call dump2d('bk_start_in',       prev_in)
  call dump2d('bk_start_future',   bucket_depth(:, :, future))
  call dump2d('bk_start_current',  bucket_depth(:, :, current))

  ! common inputs (shared by both firings)
  call dump2d('bk_cond', depth_change_cond)
  call dump2d('bk_conv', depth_change_conv)
  call dump2d('bk_lh',   depth_change_lh)
  call dump2d('bk_land', merge(1.0, 0.0, land))
  call dump_scalar('bk_max', max_bucket_depth_land)
  call dump_scalar('bk_robert', robert_bucket)
  call dump_scalar('bk_raw', raw_bucket)

  write(*, *) 'BUCKET_DUMP_DONE step future min/max:', &
       minval(bucket_depth(:, :, :)), maxval(bucket_depth(:, :, :))

contains

  ! The verbatim idealized_moist_phys L1401-1428 body. All names below are host-
  ! associated from the program above, exactly the environment the block has in
  ! idealized_moist_phys (bucket_depth, dt_bucket, filt, previous/current/future,
  ! robert_bucket, raw_bucket, depth_change_*, land, max_bucket_depth_land).
  subroutine bucket_step_body()
    include "bucket_stepping_body.inc"
  end subroutine bucket_step_body

  ! flatten a (nx,ny) field to length-n row-major and dump (matches read_dumps.py)
  subroutine dump2d(name, field)
    character(len=*), intent(in) :: name
    real, intent(in) :: field(nx, ny)
    real :: flat(n)
    integer :: ii, jj
    do jj = 1, ny
      do ii = 1, nx
        flat((jj-1)*nx + ii) = field(ii, jj)
      end do
    end do
    call jsca_dump_1d(name, flat)
  end subroutine dump2d

  subroutine dump_scalar(name, val)
    character(len=*), intent(in) :: name
    real, intent(in) :: val
    real :: one(1)
    one(1) = val
    call jsca_dump_1d(name, one)
  end subroutine dump_scalar

end program dump_bucket_stepping_reference
