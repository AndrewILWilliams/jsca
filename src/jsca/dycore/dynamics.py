"""Assembly of the dry spectral dynamical-core tendency computation.

This composes the fixture-validated kernels into the per-step spectral tendencies
of Isca's ``spectral_dynamics`` (F90 L845-910): from the current spectral state
(vorticity ``vors``, divergence ``divs``, temperature ``ts``, log surface
pressure ``ln_ps``) plus optional physics forcing, it forms
``(dt_vors, dt_divs, dt_ts, dt_ln_ps)`` — the tendencies handed to the leapfrog.

Call order (exactly the Fortran):

1. spectral -> grid: ``u, v`` from vor/div; ``T``; ``ps = exp(ln ps)``;
   grid vorticity/divergence.
2. ``pressure_variables`` + ``compute_pressure_gradient``.
3. ``four_in_one`` (pressure-gradient force, adiabatic heating, vertical velocity,
   surface-pressure tendency) and ``compute_geopotential``.
4. vertical advection of ``u, v, T``; horizontal advection of ``T``.
5. absolute-vorticity flux ``(vorg + f)`` into the momentum tendencies; convert
   to spectral vor/div tendencies; subtract the Laplacian of geopotential + KE.
6. semi-implicit correction; spectral (hyper)diffusion.

Layout (the one subtlety): the column kernels (``four_in_one``, vertical
advection, ``press_and_geopot``) and the semi-implicit / damping operators use
**level-last** arrays — grid ``(nlat, nlon, K)``, spectral ``(m, n, K)``. The
spherical transforms and vector operators use **level-leading** ``(K, nlat, nlon)``
/ ``(K, m, n)``. The prognostic state is kept level-last and moved to level-leading
only across transform calls (``_to_lead`` / ``_to_last``).

Validation: at rest (no winds, isothermal, uniform surface pressure, no forcing)
every tendency is identically zero — rest is an exact steady state of the
adiabatic dynamics — which pins the whole assembly (a mis-placed transpose or
sign breaks it). See ``tests/test_dynamics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from jsca import constants
from jsca.dycore.implicit import (
    ImplicitParams,
    implicit_correction,
    implicit_init,
)
from jsca.dycore.press_and_geopot import compute_geopotential, pressure_variables
from jsca.dycore.spectral_damping import (
    SpectralDamping,
    compute_spectral_damping,
    compute_spectral_damping_div,
    compute_spectral_damping_vor,
    spectral_damping_init,
)
from jsca.dycore.spectral_dynamics import four_in_one
from jsca.dycore.vert_advection import ADVECTIVE_FORM, SECOND_CENTERED, vert_advection
from jsca.dycore.vert_coordinate import compute_vert_coord
from jsca.grid.spectral import SpectralGrid, Truncation, laplacian_eigenvalues, total_wavenumber
from jsca.grid.transforms import (
    build_transforms,
    compute_laplacian,
    compute_pressure_gradient,
    grid_to_spectral,
    horizontal_advection,
    spectral_to_grid,
    uv_grid_from_vor_div,
    vor_div_from_uv_grid,
)

Array = jnp.ndarray


@dataclass(frozen=True)
class DynamicsParams:
    """Everything the tendency computation needs, built once (host-side)."""

    transforms: object  # TransformParams
    implicit: ImplicitParams
    damping: SpectralDamping
    pk: Array
    bk: Array
    dpk: Array
    dbk: Array
    coriolis: Array  # (nlat,)  2 omega sin(lat)
    surf_geopotential: Array  # (nlat, nlon)
    num_levels: int
    vert_difference_option: str
    use_implicit: bool
    uv_vert_advect_scheme: int
    t_vert_advect_scheme: int


def build_dynamics_params(
    num_fourier: int,
    nlat: int,
    nlon: int,
    num_levels: int,
    *,
    radius: float = constants.RADIUS,
    reference_sea_level_press: float = 101325.0,
    vert_coord_option: str = "even_sigma",
    # uneven_sigma / hybrid stretching (ignored by even_sigma); defaults mirror
    # compute_vert_coord / Isca's spectral_dynamics_nml.
    scale_heights: float = 4.0,
    surf_res: float = 0.1,
    exponent: float = 2.5,
    p_press: float = 0.1,
    p_sigma: float = 0.3,
    vert_difference_option: str = "simmons_and_burridge",
    alpha_implicit: float = 0.5,
    use_implicit: bool = True,
    ref_temperature_implicit: float = 300.0,
    damping_option: str = "resolution_dependent",
    damping_coeff: float = 1.15740741e-4,  # (0.1 day)^-1, Isca default
    damping_order: int = 2,  # del^4 (Isca default; damping ~ eigen^2 ~ l^4)
    surf_geopotential: np.ndarray | None = None,
    pk: np.ndarray | None = None,
    bk: np.ndarray | None = None,
) -> DynamicsParams:
    """Build :class:`DynamicsParams` — the Python analogue of ``spectral_dynamics_init``.

    ``pk``/``bk`` (the ``num_levels+1`` hybrid coefficients) may be passed
    explicitly to override ``vert_coord_option`` — needed for Frierson's
    ``vert_coord_option='input'`` (the 25-level table from ``frierson_test_case``).
    """
    trunc = Truncation(num_fourier)
    grid = SpectralGrid(trunc, nlat=nlat, nlon=nlon, radius=radius)
    params, gg = build_transforms(grid)

    # pk/bk/dpk/dbk are kept as NumPy (static coordinate tables). Storing them as
    # concrete arrays keeps ``pressure_variables``' top-level ``bool(pk[0] == 0)``
    # check concrete under jit (a closed-over jnp array is lifted to a tracer).
    if pk is None or bk is None:
        pk, bk = compute_vert_coord(
            vert_coord_option, num_levels, scale_heights=scale_heights, surf_res=surf_res,
            exponent=exponent, p_press=p_press, p_sigma=p_sigma,
            reference_press=reference_sea_level_press,
        )
    else:
        pk, bk = np.asarray(pk, dtype=float), np.asarray(bk, dtype=float)
    dpk, dbk = pk[1:] - pk[:-1], bk[1:] - bk[:-1]

    coriolis = jnp.asarray(2.0 * constants.OMEGA * gg.sin_lat)  # (nlat,)
    surf_geo = (
        jnp.zeros((nlat, nlon)) if surf_geopotential is None else jnp.asarray(surf_geopotential)
    )

    eigen = -laplacian_eigenvalues(trunc, radius)  # positive l(l+1)/a^2, (M+1, N+1)
    wavenumber = total_wavenumber(trunc)  # l = m + n
    num_total_wavenumbers = trunc.num_spherical - 1  # triangular truncation

    imp = implicit_init(
        pk, bk, jnp.full(num_levels, ref_temperature_implicit), reference_sea_level_press,
        num_total_wavenumbers, jnp.asarray(eigen), jnp.asarray(wavenumber),
        alpha_implicit, vert_difference_option,
    )
    damping = spectral_damping_init(
        jnp.asarray(eigen), damping_coeff, damping_order, damping_option
    )

    return DynamicsParams(
        transforms=params, implicit=imp, damping=damping,
        pk=pk, bk=bk, dpk=dpk, dbk=dbk, coriolis=coriolis, surf_geopotential=surf_geo,
        num_levels=num_levels, vert_difference_option=vert_difference_option,
        use_implicit=use_implicit,
        uv_vert_advect_scheme=SECOND_CENTERED, t_vert_advect_scheme=SECOND_CENTERED,
    )


def _to_lead(x: Array) -> Array:
    """(m, n, K) or (nlat, nlon, K) -> (K, ...) for the transforms/spherical ops."""
    return jnp.moveaxis(x, -1, 0)


def _to_last(x: Array) -> Array:
    """(K, ...) -> (..., K) back to the column / implicit / damping layout."""
    return jnp.moveaxis(x, 0, -1)


def compute_tendencies(
    p: DynamicsParams,
    vors: Array,
    divs: Array,
    ts: Array,
    ln_ps: Array,
    delta_t: float,
    wave_matrix: Array,
    previous: int,
    current: int,
    phys_dt_ug: Array | None = None,
    phys_dt_vg: Array | None = None,
    phys_dt_tg: Array | None = None,
    return_diagnostics: bool = False,
) -> tuple[Array, Array, Array, Array]:
    """Spectral dynamical tendencies for one step — port of ``spectral_dynamics``
    L845-910 (dry). ``vors``/``divs``/``ts`` are ``(m, n, K, T)`` time-level stacks,
    ``ln_ps`` is ``(m, n, T)``; ``previous``/``current`` index the last axis.
    ``phys_dt_*`` are optional grid ``(nlat, nlon, K)`` physics tendencies (default
    adiabatic). Returns ``(dt_vors, dt_divs, dt_ts, dt_ln_ps)``, damped and
    implicitly corrected.

    ``return_diagnostics=True`` additionally returns a fifth element
    ``(wg, ug, vg, p_half)`` at the *current* level — the interface vertical mass
    flux and current-level grid winds / half pressures that the moist grid-tracer
    advection (:func:`jsca.dycore.update_grid_tracer`) consumes, exactly as Isca's
    ``update_tracers`` uses ``wg`` and ``ug/vg(current)``. The dry path (flag
    False) is unchanged.
    """
    tf = p.transforms
    vc, dc, tc = vors[..., current], divs[..., current], ts[..., current]  # (m, n, K)
    lpc = ln_ps[..., current]  # (m, n)

    # 1. spectral -> grid (level-leading K, nlat, nlon)
    vorg_l = spectral_to_grid(tf, _to_lead(vc))
    divg_l = spectral_to_grid(tf, _to_lead(dc))
    ug_l, vg_l = uv_grid_from_vor_div(tf, _to_lead(vc), _to_lead(dc))
    tg_l = spectral_to_grid(tf, _to_lead(tc))
    ln_psg = spectral_to_grid(tf, lpc)  # (nlat, nlon)
    psg = jnp.exp(ln_psg)

    # column layout (nlat, nlon, K)
    ug, vg, tg = _to_last(ug_l), _to_last(vg_l), _to_last(tg_l)
    vorg, divg = _to_last(vorg_l), _to_last(divg_l)

    vdo = p.vert_difference_option
    p_half, ln_p_half, p_full, ln_p_full = pressure_variables(p.pk, p.bk, psg, vdo)
    dx_psg, dy_psg = compute_pressure_gradient(tf, lpc, psg)  # (nlat, nlon)

    z = jnp.zeros_like(tg)
    dt_ug = z if phys_dt_ug is None else phys_dt_ug
    dt_vg = z if phys_dt_vg is None else phys_dt_vg
    dt_tg = z if phys_dt_tg is None else phys_dt_tg
    dt_psg = jnp.zeros_like(psg)

    # 3. four_in_one (pressure-gradient force, adiabatic heating, vertical velocity)
    dt_psg, dt_tg, dt_ug, dt_vg, wg, wg_full = four_in_one(
        divg, ug, vg, tg, psg, ln_p_half, ln_p_full, p_full,
        dx_psg, dy_psg, dt_psg, dt_tg, dt_ug, dt_vg, p.dpk, p.dbk, p.bk, vdo,
    )
    phig_full, _phig_half = compute_geopotential(
        p.pk, tg, ln_p_half, ln_p_full, p.surf_geopotential
    )

    dt_ln_ps = grid_to_spectral(tf, dt_psg / psg)  # (m, n)

    # 4. vertical advection of u, v, T (ADVECTIVE_FORM), then horizontal advection of T
    dp = p_half[..., 1:] - p_half[..., :-1]
    dt_ug = dt_ug + vert_advection(delta_t, wg, dp, ug, p.uv_vert_advect_scheme, ADVECTIVE_FORM)
    dt_vg = dt_vg + vert_advection(delta_t, wg, dp, vg, p.uv_vert_advect_scheme, ADVECTIVE_FORM)
    dt_tg = dt_tg + vert_advection(delta_t, wg, dp, tg, p.t_vert_advect_scheme, ADVECTIVE_FORM)

    dt_tg_l = horizontal_advection(tf, _to_lead(tc), ug_l, vg_l, _to_lead(dt_tg))
    dt_ts = _to_last(grid_to_spectral(tf, dt_tg_l))  # (m, n, K)

    # 5. absolute-vorticity flux, then vor/div tendencies; subtract Laplacian(phi + KE)
    abs_vort = vorg + p.coriolis[:, None, None]
    dt_ug = dt_ug + abs_vort * vg
    dt_vg = dt_vg - abs_vort * ug
    dt_vors_l, dt_divs_l = vor_div_from_uv_grid(tf, _to_lead(dt_ug), _to_lead(dt_vg))
    dt_vors, dt_divs = _to_last(dt_vors_l), _to_last(dt_divs_l)

    phig_plus_ke = phig_full + 0.5 * (ug**2 + vg**2)
    lap = compute_laplacian(tf, grid_to_spectral(tf, _to_lead(phig_plus_ke)))
    dt_divs = dt_divs - _to_last(lap)

    # 6. semi-implicit correction, then spectral damping (on the previous level)
    if p.use_implicit:
        dt_divs, dt_ts, dt_ln_ps = implicit_correction(
            p.implicit, wave_matrix, delta_t, dt_divs, dt_ts, dt_ln_ps,
            divs, ts, ln_ps, previous, current,
        )
    dt_vors = compute_spectral_damping_vor(p.damping, vors[..., previous], dt_vors, delta_t)
    dt_divs = compute_spectral_damping_div(p.damping, divs[..., previous], dt_divs, delta_t)
    dt_ts = compute_spectral_damping(p.damping, ts[..., previous], dt_ts, delta_t)

    # Triangular-truncate the prognostic tendencies to l = m + n <= M. Isca's
    # trans_grid_to_spherical truncates the prognostic tendencies (dt_ts, dt_ln_ps
    # and, via the truncated geopotential+KE, dt_divs) to the triangle by default,
    # whereas jsca's grid_to_spectral keeps the l = M+1 storage diagonal (needed
    # for the d/dmu derivative paths). Left untruncated, the l = M+1 modes are
    # unphysical grid-scale content that nothing damps: at high resolution / large
    # dt they grow ~exponentially and blow the model up (the T42 dt=600
    # instability). Zeroing them here restores per-step agreement with Isca to
    # machine precision. vor/div are already triangular from vor_div_from_uv_grid,
    # but dt_divs picks up the l=M+1 row from the Laplacian of (phi + KE).
    mp = p.transforms.mask_prognostic  # (m, n), True where l <= M
    dt_vors = jnp.where(mp[..., None], dt_vors, 0.0)
    dt_divs = jnp.where(mp[..., None], dt_divs, 0.0)
    dt_ts = jnp.where(mp[..., None], dt_ts, 0.0)
    dt_ln_ps = jnp.where(mp, dt_ln_ps, 0.0)

    if return_diagnostics:
        return dt_vors, dt_divs, dt_ts, dt_ln_ps, (wg, ug, vg, p_half)
    return dt_vors, dt_divs, dt_ts, dt_ln_ps


def rest_state(p: DynamicsParams, nlat: int, nlon: int, temperature: float, surface_press: float):
    """A resting isothermal state: zero vor/div, uniform ``T`` and ``ps``.

    Returns ``(vors, divs, ts, ln_ps)`` single-level-time spectral arrays
    ``(m, n, K)`` / ``(m, n)``; stack along a new last axis for the leapfrog.
    """
    tf = p.transforms
    k = p.num_levels
    tg = jnp.full((k, nlat, nlon), temperature)
    ln_psg = jnp.full((nlat, nlon), float(np.log(surface_press)))
    ts = _to_last(grid_to_spectral(tf, tg))  # (m, n, K)
    ln_ps = grid_to_spectral(tf, ln_psg)  # (m, n)
    zero = jnp.zeros_like(ts)
    return zero, zero, ts, ln_ps
