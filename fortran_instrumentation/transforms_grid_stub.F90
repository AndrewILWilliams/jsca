!! transforms_mod grid-side stub for the water_borrowing and global_integral
!! fixture drivers.
!!
!! water_borrowing.F90 pulls get_grid_domain; global_integral.F90 pulls
!! area_weighted_global_mean. This stub supplies both:
!!   * get_grid_domain returns the full serial grid domain (single PE):
!!     is=1, ie=num_lon, js=1, je=num_lat.
!!   * area_weighted_global_mean *replicates verbatim* the source it wraps,
!!     $ISCA_SRC/src/atmos_spectral/tools/transforms.F90 (function
!!     area_weighted_global_mean, lines 1059-1077):
!!         weighted(:,j) = wts_lat(j)*field(:,j)
!!         mean = sum(weighted) / (sum(wts_lat) * num_lon)
!!     using the *real* Gaussian latitude weights passed to the init below (which
!!     the driver obtains from the real gauss_and_legendre compute_gaussian) — no
!!     reinvented numerics, only real values and the transposed formula.

module transforms_mod
implicit none
private
public :: transforms_grid_stub_init, get_grid_domain, area_weighted_global_mean

integer, save :: is = 1, ie = 0, js = 1, je = 0, num_lon = 0, num_lat = 0
real, allocatable, save :: wts_lat(:)

contains

subroutine transforms_grid_stub_init(num_lon_in, num_lat_in, wts_lat_in)
  integer, intent(in) :: num_lon_in, num_lat_in
  real,    intent(in) :: wts_lat_in(:)
  num_lon = num_lon_in
  num_lat = num_lat_in
  is = 1; ie = num_lon; js = 1; je = num_lat
  if (allocated(wts_lat)) deallocate(wts_lat)
  allocate(wts_lat(num_lat))
  wts_lat = wts_lat_in
end subroutine transforms_grid_stub_init

subroutine get_grid_domain(is_out, ie_out, js_out, je_out)
  integer, intent(out) :: is_out, ie_out, js_out, je_out
  is_out = is; ie_out = ie; js_out = js; je_out = je
end subroutine get_grid_domain

function area_weighted_global_mean(field) result(mean)
  real, intent(in), dimension(:,:) :: field
  real :: mean
  real, dimension(size(field,1), size(field,2)) :: weighted
  real :: global_sum_of_wts
  integer :: j
  do j = 1, size(field,2)
    weighted(:,j) = wts_lat(j)*field(:,j)
  end do
  global_sum_of_wts = sum(wts_lat)
  mean = sum(weighted)/(global_sum_of_wts*num_lon)
end function area_weighted_global_mean

end module transforms_mod
