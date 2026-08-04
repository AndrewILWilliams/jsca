!! Minimal mpp_domains_mod stub. global_integral.F90 imports mpp_global_field
!! (only: mpp_global_field) but never calls it in the serial path exercised
!! here; this supplies the symbol so the `use` resolves. No-op, no numerics.

module mpp_domains_mod
implicit none
private
public :: mpp_global_field

contains

subroutine mpp_global_field(local, global)
  real, intent(in)  :: local(:,:)
  real, intent(out) :: global(:,:)
  global = local
end subroutine mpp_global_field

end module mpp_domains_mod
