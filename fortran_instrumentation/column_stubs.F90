!! Minimal link-only stubs for building the single-column initial-condition dumper
!! (dump_column_init_reference.F90) against the UNMODIFIED Isca
!! src/atmos_column/column_initialize_fields.F90.
!!
!! column_initialize_fields uses only two things outside constants_mod/fms_mod:
!!   * spec_mpp_mod::get_grid_domain  -- to size the local array (here: 1 column)
!!   * column_grid_mod::area_weighted_global_mean  -- ONLY to feed a diagnostic
!!     print of the mean surface pressure (its value never touches any dumped
!!     output). Per the CLAUDE.md stub rule ("stubs may contain logging only, never
!!     reimplemented numerics"), this returns a harmless placeholder: it is not
!!     used in any golden quantity, so no numerics are being faked.
!!
!! These are separate module definitions from spec_mpp_stub.F90 / the transforms
!! stubs; link THIS file (not those) for the column-init dumper.

module spec_mpp_mod
implicit none
private
public :: get_grid_domain
! single-column compute domain (lon_max = lat_max = 1)
contains
  subroutine get_grid_domain(is, ie, js, je)
    integer, intent(out) :: is, ie, js, je
    is = 1; ie = 1; js = 1; je = 1
  end subroutine get_grid_domain
end module spec_mpp_mod


module column_grid_mod
implicit none
private
public :: area_weighted_global_mean
contains
  ! Link-only: only feeds the "mean surface pressure" print in
  ! column_initialize_fields; never enters a dumped result.
  function area_weighted_global_mean(field)
    real :: area_weighted_global_mean
    real, intent(in), dimension(:,:) :: field
    area_weighted_global_mean = sum(field) / max(1, size(field))
  end function area_weighted_global_mean
end module column_grid_mod
