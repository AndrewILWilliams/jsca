"""A-grid horizontal (van Leer) advection on the sphere.

Faithful port of Isca's ``src/atmos_spectral/model/fv_advection.F90`` (module
``fv_advection_mod``, subroutine ``a_grid_horiz_advection``). This is the
Lin--Rood-style finite-volume advection used for tracer transport on the
Gaussian grid: a directionally-split scheme where the full-step zonal flux uses
a half-step *meridionally* advanced field and the full-step meridional flux uses
a half-step *zonally* advanced field (``advection_sphere_3d``, F90 L238-285).

Serial full-domain port
-----------------------
jsca runs the full (undecomposed) grid on one device, i.e. the Fortran
``js == 1``, ``je == ny``, ``is == 1``, ``ie == nx`` branch. In that regime the
``mpp_update_domains`` halo exchanges (F90 L161-162, L259) are no-ops — the
meridional halos are filled *entirely* by the explicit pole-reflection code
(F90 L164-178, L266-278), and the zonal direction is periodic and handled by
index wrapping. This port reproduces exactly that serial behaviour; it is not a
domain-decomposed implementation.

Pole reflection (F90 L153-156, L164-178): the point "just across" the pole at
longitude index ``i`` is the interior point at longitude ``i + nx/2`` (Fortran
``ii(i)``), i.e. the antipodal meridian. ``q`` reflects with the same sign; the
meridional wind ``v`` reflects with a sign flip. With ``nx`` even (all Isca
grids) a longitude shift of ``nx/2`` is ``jnp.roll(row, -(nx // 2), axis=-1)``.

Layout
------
FV advection is a *horizontal* operator (the vertical level is a passive loop,
F90 e.g. L183/L355/L402), so — unlike the column-physics modules — fields here
follow the transform convention ``(..., nlat, nlon)`` with latitude on axis
``-2`` and longitude (periodic) on axis ``-1``; the level and any other axes are
leading batch dimensions. The single Fortran routine and its 2-D convenience
wrapper (``a_grid_horiz_advection_2d``, F90 L211-234) collapse to this one
function. Fortran storage is ``(lon, lat, lev)``; fixtures move axes at the
boundary. Latitudes run south -> north, cell edges ``yy`` (``lat_edges``) has
``ny + 1`` entries with the poles at ``+/- pi/2``.

Faithful subtleties preserved
------------------------------
* Two *different* fractional-Courant conventions: the semi-Lagrangian inner
  half-steps (``semi_x``, F90 L400) use ``b - floor(b)`` (fraction in ``[0, 1)``),
  while the flux-form ``vanleer_x`` (F90 L345) uses ``b - int(b)`` — Fortran
  ``int`` truncates toward zero, giving a fraction in ``(-1, 1)``. Reproduced as
  ``jnp.floor`` vs ``jnp.trunc`` respectively.
* ``find_cell_x`` (F90 L441-458) wraps the upstream index into ``[1, nx]`` with a
  *single* add/subtract of ``nx`` (not a full modulo). Faithful for
  ``|floor(b)| <= 1``, i.e. Courant magnitude ``< 2`` in the semi-Lagrangian
  step; reproduced verbatim.
* ``integer_flux_x`` (F90 L498-533): the whole-cells-crossed flux for
  ``|Courant| > 1`` (common near the poles where ``cos(lat) -> 0``). Computed
  here in closed form via a tiled prefix sum (exact for ``|int(b)| < nx``). The
  Fortran gates it per latitude row with ``maxval(abs(b)) > 1`` reduced over
  longitude *and level* (F90 L349); this port reduces the gate over longitude
  per slice only. The two differ only for a grid point whose Courant number is
  *exactly* ``+/- 1.0`` while another level at the same latitude exceeds 1 — a
  measure-zero case that never arises for the random fixture data (agreement is
  to machine precision).
* ``sign(1.0, x)`` (F90 L364, L488, L554) returns ``+1`` for ``x >= 0`` and
  ``-1`` for ``x < 0``; rendered as ``where(x < 0, -1.0, 1.0)``.
* The unused ``dyyy`` array and the ``solid_body`` check-out routine (F90 L564)
  are not ported. ``monotone`` is Isca's default ``.true.`` (F90 L40).

Fixtures: ``tests/fixtures/fv_advection_reference.npz`` from
``fortran_instrumentation/dump_fv_advection_reference.F90`` (real Fortran).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jsca import constants

Array = jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FvAdvectionParams:
    """Precomputed grid metrics for :func:`a_grid_horiz_advection`.

    All length-``L`` arrays are indexed by latitude and broadcast against the
    ``-2`` axis. Radius scaling (F90 L103-106) is already applied to ``dy_pad``
    and ``dyy``; ``dy_plus`` / ``dy_minus`` are dimensionless ratios.
    """

    c: Array  # (ny,)     cos(cell-centre lat)
    cc: Array  # (ny+1,)  cos(cell-edge lat)
    dy_pad: Array  # (ny+2,)  radius*dy(j), j = 0 .. ny+1 (halo-padded cell size)
    dyy: Array  # (ny+1,)   radius*dyy(j), j = 1 .. ny+1 (centre-to-centre)
    dy_plus: Array  # (ny+2,)  dy(j)/(dy(j)+dy(j+1)),   j = 0 .. ny+1
    dy_minus: Array  # (ny+2,) dy(j)/(dy(j-1)+dy(j)),   j = 0 .. ny+1
    dx: float  # zonal grid spacing at the equator (radius included)
    num_lon: int
    num_lat: int

    def tree_flatten(self):
        children = (self.c, self.cc, self.dy_pad, self.dyy, self.dy_plus, self.dy_minus)
        aux = (self.dx, self.num_lon, self.num_lat)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)


def fv_advection_init(
    num_lon: int,
    lat_edges: np.ndarray,
    degrees_lon: float = 360.0,
    radius: float | None = None,
) -> FvAdvectionParams:
    """Build grid metrics — port of ``fv_advection_init`` (F90 L59-122).

    ``lat_edges`` are the ``ny + 1`` cell-edge latitudes in radians, south to
    north (Fortran ``yy_in``). ``degrees_lon`` is the zonal span (360 for a full
    sphere). Host-side (NumPy) — done once.
    """
    if radius is None:
        radius = constants.RADIUS
    yy = np.asarray(lat_edges, dtype=np.float64)
    ny = yy.size - 1
    nx = int(num_lon)

    y = 0.5 * (yy[1:] + yy[:-1])  # cell centres (ny,), F90 L84
    c = np.cos(y)  # F90 L86
    cc = np.cos(yy)  # F90 L88

    # dy(j), the cell size, indexed j = -1 .. ny+2 (F90 L79, L90-94); store the
    # slice j = 0 .. ny+1 that the compute path needs. dy(0)=dy(1), dy(ny+1)=dy(ny).
    dy_core = yy[1:] - yy[:-1]  # dy(1..ny)
    dy = np.empty(ny + 4)  # index p -> j = p - 1  (j = -1 .. ny+2)
    dy[2 : ny + 2] = dy_core  # j = 1..ny
    dy[0] = dy_core[1]  # dy(-1) = dy(2)
    dy[1] = dy_core[0]  # dy(0)  = dy(1)
    dy[ny + 2] = dy_core[ny - 1]  # dy(ny+1) = dy(ny)
    dy[ny + 3] = dy_core[ny - 2]  # dy(ny+2) = dy(ny-1)

    # dyy(j), centre-to-centre distance, indexed j = 1 .. ny+1 (F90 L96-98)
    dyy = np.empty(ny + 1)  # index m -> j = m + 1
    dyy[1:ny] = y[1:] - y[:-1]  # dyy(2..ny)
    dyy[0] = 2.0 * (y[0] - yy[0])  # dyy(1)
    dyy[ny] = 2.0 * (yy[ny] - y[ny - 1])  # dyy(ny+1)

    # dy_plus/dy_minus (F90 L100-101), j = 0 .. ny+1 — dimensionless ratios
    j = np.arange(ny + 2)  # j = 0 .. ny+1  ->  dy index p = j + 1
    dy_plus = dy[j + 1] / (dy[j + 1] + dy[j + 2])
    dy_minus = dy[j + 1] / (dy[j] + dy[j + 1])

    dy_pad = radius * dy[1 : ny + 3]  # radius*dy(0..ny+1), F90 L105
    dyy = radius * dyy  # F90 L106
    dx = (degrees_lon / 360.0) * 2.0 * np.pi * radius / float(nx)  # F90 L108

    return FvAdvectionParams(
        c=jnp.asarray(c),
        cc=jnp.asarray(cc),
        dy_pad=jnp.asarray(dy_pad),
        dyy=jnp.asarray(dyy),
        dy_plus=jnp.asarray(dy_plus),
        dy_minus=jnp.asarray(dy_minus),
        dx=float(dx),
        num_lon=nx,
        num_lat=ny,
    )


# --------------------------------------------------------------------------- #
# halo construction (pole reflection + periodic longitude)
# --------------------------------------------------------------------------- #


def _reflect(row: Array, nx: int) -> Array:
    """Antipodal-meridian value: ``result[i] = row[(i + nx/2) mod nx]`` (F90 ii)."""
    return jnp.roll(row, -(nx // 2), axis=-1)


def _qpad(q: Array, nx: int) -> Array:
    """Pad ``q`` (..., ny, nx) with two pole-reflected halo rows each side.

    Returns (..., ny+4, nx); padded index ``p`` maps to Fortran latitude
    ``j = p - 1`` (so ``j = -1 .. ny+2``). Scalar reflection, no sign flip
    (F90 L164-178 for ``qx``; L266-278 for ``q1``).
    """
    ny = q.shape[-2]
    south0 = _reflect(q[..., 0, :], nx)  # q(j=1)  -> halo j=0
    southm1 = _reflect(q[..., 1, :], nx)  # q(j=2)  -> halo j=-1
    north0 = _reflect(q[..., ny - 1, :], nx)  # q(j=ny)   -> halo j=ny+1
    northp1 = _reflect(q[..., ny - 2, :], nx)  # q(j=ny-1) -> halo j=ny+2
    return jnp.concatenate(
        [
            southm1[..., None, :],
            south0[..., None, :],
            q,
            north0[..., None, :],
            northp1[..., None, :],
        ],
        axis=-2,
    )


def _vpad(v: Array, nx: int) -> Array:
    """Pad ``v`` (..., ny, nx) with one sign-flipped pole-reflected row each side.

    Returns (..., ny+2, nx); padded index ``p`` maps to Fortran latitude
    ``j = p`` (so ``j = 0 .. ny+1``). Meridional wind flips sign across the pole
    (F90 L166, L174).
    """
    ny = v.shape[-2]
    south = -_reflect(v[..., 0, :], nx)  # -v(j=1) -> halo j=0
    north = -_reflect(v[..., ny - 1, :], nx)  # -v(j=ny) -> halo j=ny+1
    return jnp.concatenate([south[..., None, :], v, north[..., None, :]], axis=-2)


# --------------------------------------------------------------------------- #
# zonal (x) operators — longitude is axis -1, periodic
# --------------------------------------------------------------------------- #


def _find_cell0(b: Array, nx: int) -> Array:
    """Upstream cell (0-based lon index) — port of ``find_cell_x`` (F90 L441-458).

    ``ii(i) = i - 1 - floor(b)`` wrapped into ``[1, nx]`` with a single
    add/subtract of ``nx``; returned 0-based (``ii - 1``).
    """
    i0 = jnp.arange(nx)
    ii = i0 - jnp.floor(b)  # base i-1 (0-based) minus floor(b)
    ii = jnp.where(ii > nx, ii - nx, ii)
    ii = jnp.where(ii < 1, ii + nx, ii)
    return (ii - 1).astype(jnp.int32)


def _slope_x(q: Array, nx: int) -> Array:
    """Monotonic van Leer slope in longitude — ``slope_x`` (F90 L462-494)."""
    grad = q - jnp.roll(q, 1, axis=-1)  # grad(i) = q(i) - q(i-1)
    slope = 0.5 * (jnp.roll(grad, -1, axis=-1) + grad)  # (grad(i+1)+grad(i))/2
    qm1 = jnp.roll(q, 1, axis=-1)
    qp1 = jnp.roll(q, -1, axis=-1)
    q_min = jnp.minimum(jnp.minimum(qm1, q), qp1)
    q_max = jnp.maximum(jnp.maximum(qm1, q), qp1)
    limited = jnp.minimum(jnp.minimum(jnp.abs(slope), 2.0 * (q - q_min)), 2.0 * (q_max - q))
    return jnp.where(slope < 0.0, -1.0, 1.0) * limited


def _integer_flux_x(b: Array, q: Array, nx: int) -> Array:
    """Whole-cells-crossed flux for ``|Courant| > 1`` — ``integer_flux_x`` (F90 L498-533).

    For edge ``i`` and ``m = int(b(i))`` (truncation): ``m`` cells upstream summed
    for ``m > 0``, ``-|m|`` cells downstream for ``m < 0``, ``0`` for ``m = 0``.
    Evaluated in closed form with a tiled prefix sum (exact for ``|m| < nx``).
    """
    m = jnp.trunc(b).astype(jnp.int32)
    qt = jnp.concatenate([q, q], axis=-1)  # tile longitude
    zero = jnp.zeros_like(q[..., :1])
    prefix = jnp.concatenate([zero, jnp.cumsum(qt, axis=-1)], axis=-1)  # P[t]=sum_{s<t}
    e = jnp.arange(nx)  # 0-based edge index, broadcasts on axis -1
    # m>=0: flux = P[e+nx] - P[e-m+nx];  m<0: flux = P[e] - P[e-m]
    hi = jnp.where(m >= 0, e + nx, e).astype(jnp.int32)
    lo = jnp.where(m >= 0, e - m + nx, e - m).astype(jnp.int32)
    return jnp.take_along_axis(prefix, hi, axis=-1) - jnp.take_along_axis(prefix, lo, axis=-1)


def _semi_x(ua: Array, q: Array, dt: float, p: FvAdvectionParams) -> Array:
    """Semi-Lagrangian half-step increment in longitude — ``semi_x_3d`` (F90 L379-415)."""
    nx = p.num_lon
    b = ua * dt / (p.dx * p.c[:, None])  # Courant (F90 L391)
    floor_b = jnp.floor(b)
    ii = jnp.arange(nx) - floor_b
    ii = jnp.where(ii > nx, ii - nx, ii)
    ii = jnp.where(ii < 1, ii + nx, ii)
    i_left = (ii - 1).astype(jnp.int32)  # 0-based
    i_right = jnp.mod(ii.astype(jnp.int32), nx)  # ii+1 wrapped, 0-based
    bb = b - floor_b  # fraction in [0, 1)
    q_left = jnp.take_along_axis(q, i_left, axis=-1)
    q_right = jnp.take_along_axis(q, i_right, axis=-1)
    return bb * q_left + (1.0 - bb) * q_right - q  # F90 L411


def _vanleer_x(dq_dt: Array, uc: Array, q: Array, dt: float, p: FvAdvectionParams) -> Array:
    """Flux-form van Leer advection in longitude — ``vanleer_x_3d`` (F90 L330-375)."""
    nx = p.num_lon
    b = uc * dt / (p.dx * p.c[:, None])  # edge Courant (F90 L343)
    bb = b - jnp.trunc(b)  # int() truncation (F90 L345)
    int_flux = _integer_flux_x(b, q, nx)
    row_gate = jnp.max(jnp.abs(b), axis=-1, keepdims=True) > 1.0  # (F90 L349)
    int_flux = jnp.where(row_gate, int_flux, 0.0)
    s = _slope_x(q, nx)
    i_up = _find_cell0(b, nx)  # find_cell_x uses floor(b) (F90 L353)
    qq = jnp.take_along_axis(q, i_up, axis=-1)
    ss = jnp.take_along_axis(s, i_up, axis=-1)
    frac_flux = bb * (qq + 0.5 * ss * (jnp.where(bb < 0.0, -1.0, 1.0) - bb))  # F90 L364
    flux = int_flux + frac_flux  # flux(1:nx)
    flux_ip1 = jnp.roll(flux, -1, axis=-1)  # flux(2:nx+1), flux(nx+1)=flux(1) (F90 L365)
    return dq_dt - (flux_ip1 - flux) / dt  # F90 L367


# --------------------------------------------------------------------------- #
# meridional (y) operators — latitude is axis -2
# --------------------------------------------------------------------------- #


def _semi_y(va: Array, qpad: Array, dt: float, p: FvAdvectionParams) -> Array:
    """Semi-Lagrangian half-step increment in latitude — ``semi_y_3d`` (F90 L419-437).

    ``qpad`` is the halo-padded tracer (..., ny+4, nx); returns (..., ny, nx).
    """
    ny = p.num_lat
    q_jm1 = qpad[..., 1 : ny + 1, :]  # q(j-1), j = 1..ny
    q_j = qpad[..., 2 : ny + 2, :]  # q(j)
    q_jp1 = qpad[..., 3 : ny + 3, :]  # q(j+1)
    dyy_j = p.dyy[0:ny][:, None]  # dyy(j),   j = 1..ny
    dyy_jp1 = p.dyy[1 : ny + 1][:, None]  # dyy(j+1), j = 2..ny+1
    pos = va * dt * (q_jm1 - q_j) / dyy_j
    neg = va * dt * (q_j - q_jp1) / dyy_jp1
    return jnp.where(va >= 0.0, pos, neg)


def _slope_sphere(qpad: Array, p: FvAdvectionParams) -> Array:
    """Monotonic slope in latitude on the non-uniform grid — ``slope_sphere`` (F90 L537-560).

    ``qpad`` (..., ny+4, nx); returns slope (..., ny+2, nx) for j = 0 .. ny+1.
    """
    ny = p.num_lat
    q_jm1 = qpad[..., 0 : ny + 2, :]  # q(j-1), j = 0..ny+1
    q_j = qpad[..., 1 : ny + 3, :]  # q(j)
    q_jp1 = qpad[..., 2 : ny + 4, :]  # q(j+1)
    slope = (q_jp1 - q_j) * p.dy_plus[:, None] + (q_j - q_jm1) * p.dy_minus[:, None]
    q_min = jnp.minimum(jnp.minimum(q_jm1, q_j), q_jp1)
    q_max = jnp.maximum(jnp.maximum(q_jm1, q_j), q_jp1)
    limited = jnp.minimum(jnp.minimum(jnp.abs(slope), 2.0 * (q_j - q_min)), 2.0 * (q_max - q_j))
    return jnp.where(slope < 0.0, -1.0, 1.0) * limited


def _vanleer_sphere(dq_dt: Array, vc: Array, qpad: Array, dt: float, p: FvAdvectionParams) -> Array:
    """Flux-form van Leer advection in latitude on the sphere — ``vanleer_sphere_3d`` (F90 L289).

    ``vc`` (..., ny+1, nx) is the edge meridional wind (j = 1..ny+1); ``qpad``
    (..., ny+4, nx) the padded tracer. Returns updated ``dq_dt`` (..., ny, nx).
    """
    ny = p.num_lat
    s = _slope_sphere(qpad, p)  # (..., ny+2, nx), index m -> j = m (0..ny+1)
    ccj = p.cc[0 : ny + 1][:, None]  # cc(j), j = 1..ny+1
    q_jm1 = qpad[..., 1 : ny + 2, :]  # q(j-1), j = 1..ny+1
    q_j = qpad[..., 2 : ny + 3, :]  # q(j)
    s_jm1 = s[..., 0 : ny + 1, :]  # s(j-1)
    s_j = s[..., 1 : ny + 2, :]  # s(j)
    dtdy_jm1 = (dt / p.dy_pad[0 : ny + 1])[:, None]  # dtdy(j-1), j = 1..ny+1
    dtdy_j = (dt / p.dy_pad[1 : ny + 2])[:, None]  # dtdy(j)
    flux_pos = vc * ccj * (q_jm1 + 0.5 * s_jm1 * (1.0 - dtdy_jm1 * vc))  # F90 L310-311
    flux_neg = vc * ccj * (q_j - 0.5 * s_j * (1.0 + dtdy_j * vc))  # F90 L313-314
    flux = jnp.where(vc >= 0.0, flux_pos, flux_neg)  # (..., ny+1, nx)
    # no flux through the poles: flux(j=1) = flux(j=ny+1) = 0 (F90 L318-319)
    edge_keep = jnp.ones(ny + 1).at[0].set(0.0).at[ny].set(0.0)
    flux = flux * edge_keep[:, None]
    dyc = (1.0 / (p.dy_pad[1 : ny + 1] * p.c))[:, None]  # 1/(dy(j)*c(j)), j = 1..ny
    return dq_dt - dyc * (flux[..., 1 : ny + 1, :] - flux[..., 0:ny, :])  # F90 L321-323


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def _advection_sphere(
    dq_dt: Array,
    dt: float,
    qpad: Array,
    uc: Array,
    vc: Array,
    ua: Array,
    va: Array,
    p: FvAdvectionParams,
) -> Array:
    """Directionally-split cross-advection — ``advection_sphere_3d`` (F90 L238-285)."""
    nx, ny = p.num_lon, p.num_lat
    q_int = qpad[..., 2 : ny + 2, :]  # interior tracer, j = 1..ny

    q1_int = q_int + _semi_x(ua, q_int, 0.5 * dt, p)  # half x-step (F90 L253-254)
    q2_int = q_int + _semi_y(va, qpad, 0.5 * dt, p)  # half y-step (F90 L256-257)

    q1pad = _qpad(q1_int, nx)  # pole-reflect the half x-advanced field (F90 L259-278)

    dq_dt = _vanleer_x(dq_dt, uc, q2_int, dt, p)  # x-flux of y-advanced q2 (F90 L280)
    dq_dt = _vanleer_sphere(dq_dt, vc, q1pad, dt, p)  # y-flux of x-advanced q1 (F90 L282)
    return dq_dt


def a_grid_horiz_advection(
    ua: Array,
    va: Array,
    q: Array,
    dt: float,
    dq_dt: Array,
    p: FvAdvectionParams,
    flux: bool = False,
) -> Array:
    """A-grid horizontal advection tendency — port of ``a_grid_horiz_advection`` (F90 L126-207).

    ``ua`` (zonal wind), ``va`` (meridional wind), ``q`` (tracer) and ``dq_dt``
    are ``(..., nlat, nlon)`` with latitude on axis ``-2`` and periodic longitude
    on axis ``-1``; leading axes (e.g. level) are batched. Returns ``dq_dt`` with
    the advective tendency added.

    ``flux`` is the Fortran optional argument (default ``.false.``): when true the
    output is the pure flux-form divergence and the ``+ q * div`` advective
    correction (F90 L191-202) is skipped. Pass it as a static argument under jit.
    """
    nx, ny = p.num_lon, p.num_lat

    uc = 0.5 * (jnp.roll(ua, 1, axis=-1) + ua)  # zonal wind on x-edges (F90 L180-181)
    vpad = _vpad(va, nx)  # v with pole halos (F90 L158-178)
    vc = 0.5 * (vpad[..., 0 : ny + 1, :] + vpad[..., 1 : ny + 2, :])  # y-edge wind (F90 L186)
    qpad = _qpad(q, nx)  # tracer with pole halos

    if not flux:
        cc_j = p.cc[0:ny][:, None]  # cc(j),   j = 1..ny
        cc_j1 = p.cc[1 : ny + 1][:, None]  # cc(j+1), j = 1..ny
        dy_j = p.dy_pad[1 : ny + 1][:, None]  # dy(j),   j = 1..ny
        div = (vc[..., 1 : ny + 1, :] * cc_j1 - vc[..., 0:ny, :] * cc_j) / (p.c[:, None] * dy_j)
        div = div + (jnp.roll(uc, -1, axis=-1) - uc) / (p.c[:, None] * p.dx)  # F90 L197-198
        dq_dt = dq_dt + q * div  # F90 L201

    return _advection_sphere(dq_dt, dt, qpad, uc, vc, ua, va, p)
