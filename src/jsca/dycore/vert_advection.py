"""Vertical advection of grid-column fields.

Faithful port of Isca's ``src/atmos_shared/vert_advection/vert_advection.F90``
(``vert_advection_3d``). Given an advecting velocity ``w`` on the ``K + 1`` layer
interfaces and a field ``r`` on the ``K`` layers, it forms the interface fluxes
for one of several differencing schemes and returns the vertical advective
tendency ``rdt``. This is the ``vert_advection`` used by ``spectral_dynamics`` for
the vertical transport of ``u``, ``v``, ``T`` and tracers.

Layout: column physics, so the **level axis is last** — ``r``, ``dz`` are
``(..., K)`` and ``w`` is ``(..., K + 1)`` (interfaces, index 0 = top … K =
surface), matching the Fortran ``(:, :, k)`` with ``k = 1`` at the top. Leading
axes are batched.

Schemes ported (all jit/vmap/scan-safe; F90 ``select case`` at L170):

* ``SECOND_CENTERED``      — 2nd-order centered, uniform spacing (F90 L185)
* ``SECOND_CENTERED_WTS``  — 2nd-order centered, unequal spacing (F90 L173)
* ``FOURTH_CENTERED``      — 4th-order centered, uniform spacing (F90 L239)
* ``FOURTH_CENTERED_WTS``  — 4th-order centered, unequal spacing (F90 L196)
* ``FINITE_VOLUME_LINEAR`` (== ``VAN_LEER_LINEAR``) — piecewise-linear FV (F90 L277)

Both equation forms (``FLUX_FORM`` = ``-d(wr)/dt``, ``ADVECTIVE_FORM`` =
``-w d(r)/dt``; F90 L449) and the ``WEIGHTED_TENDENCY`` flag (F90 L108, drops the
``/dz``) are supported. The Held–Suarez default path is ``SECOND_CENTERED`` +
``ADVECTIVE_FORM`` (Isca namelist default ``vert_advect_* = 'second_centered'``);
Frierson advects ``sphum`` with ``FINITE_VOLUME_PARABOLIC`` (PPM).

The PPM scheme (``FINITE_VOLUME_PARABOLIC``, F90 L301-438) is ported: piecewise-
parabolic reconstruction with the unequal-spacing 4th-order edge weights, the
Colella-Woodward monotonicity limiter, and the flux integral over the departure
region **including the Courant>1 extension**. That extension (F90's ``do while``
that accumulates whole cells until the swept distance ``dt|w|`` is covered) is a
departure-point integral, so it is expressed here as a loop-free ``searchsorted``
on the cumulative-``dz`` prefix — no ``lax.while_loop`` needed — and the single
formula reduces to the plain single-cell PPM flux when ``Courant <= 1``.

**An Isca bug reproduced-around, not reproduced.** Isca's PPM Courant>1 walk for
*downward* (``w < 0``) interfaces exits on ``kk == ks`` (F90 L414) while
incrementing ``kk`` toward ``ke``, so ``kk`` runs off the end and reads
``dz(ke+1)`` **out of bounds** (undefined memory under ``-O2``). The upward
(``w >= 0``) branch exits correctly on ``kk == ks`` (F90 L387). This port clamps
the departure cell at ``ke`` — the obviously-intended behaviour — so it is
well-defined for both signs. The fixture validates every well-defined path
(all Courant for ``w >= 0``; ``|Courant| < 1`` for ``w < 0``); the buggy
``w < 0`` & ``Courant > 1`` path is not fixture-checked (Isca's output there is
garbage) and, per Frierson's sub-unity resolved vertical Courant number, is not
reached in the target run. ``FINITE_VOLUME_PARABOLIC2`` (the Lin-2003 relaxed
limiter, F90 L340-349) is still a follow-up.

Deliberately not ported here (raise ``NotImplementedError``, documented, off the
default step path):

* ``FINITE_VOLUME_PARABOLIC2`` (Lin-2003 limiter variant, F90 L340) — unused by
  Frierson (which selects plain ``finite_volume_parabolic``).
* ``mask`` (below-ground layers, F90 L199/L241) — only alters the 4th-order
  schemes adjacent to ground; ``spectral_dynamics`` never passes it.
* ``OUTFLOW_BOUNDARY`` (F90 L109) — finite-volume outflow BCs; unused by the
  dynamical core.

Boundary fluxes (non-outflow, F90 L166-167): ``flux[0] = w[0] r[0]`` and
``flux[K] = w[K] r[K-1]`` (``w`` is generally 0 there but not assumed so).

Fortran subtleties preserved
----------------------------
* ``slope_z`` (F90 L505): the limited slope is forced to **0** at the top and
  bottom layers (F90 L560), overriding the ``2*grad*dz`` end values (F90
  L542-543) whenever limiters are on (van Leer). With limiters off (the
  ``*_WTS`` 4th-order path) the end slopes keep ``2*grad*dz``.
* ``sign(1., slope)`` (F90 L551) is ``+1`` for ``slope >= 0`` — ``where(slope < 0,
  -1., 1.)``.

Fixtures: ``tests/fixtures/vert_advection_reference.npz`` and
``tests/fixtures/ppm_advection_reference.npz`` from the matching
``dump_*_reference.F90`` drivers (real Fortran).
"""

from __future__ import annotations

import jax.numpy as jnp

Array = jnp.ndarray

# scheme selectors (F90 L34-41) — values must match the Fortran integers so a
# fixture / caller may pass them through unchanged.
SECOND_CENTERED = 101
FOURTH_CENTERED = 102
FINITE_VOLUME_LINEAR = 103
FINITE_VOLUME_PARABOLIC = 104
FINITE_VOLUME_PARABOLIC2 = 105
SECOND_CENTERED_WTS = 106
FOURTH_CENTERED_WTS = 107
VAN_LEER_LINEAR = FINITE_VOLUME_LINEAR

# equation-form selectors (F90 L43)
FLUX_FORM = 201
ADVECTIVE_FORM = 202

# flags (F90 L45-46)
WEIGHTED_TENDENCY = 1
OUTFLOW_BOUNDARY = 2


def _cat(*parts: Array) -> Array:
    return jnp.concatenate(parts, axis=-1)


def _slope_z(r: Array, dz: Array, limit: bool = True, linear: bool = True) -> Array:
    """Slope of ``r`` along the level axis — port of ``slope_z`` (F90 L505-568).

    ``r``, ``dz`` are ``(..., K)``. ``limit`` applies the van-Leer monotonicity
    limiter (and zeroes the end slopes); ``linear`` selects the simple vs
    unequal-spacing interior weighting.
    """
    k = r.shape[-1]
    # grad[j] = (r[j] - r[j-1])/(dz[j] + dz[j-1]) for j = 1..K-1 (F90 L529)
    grad_body = (r[..., 1:] - r[..., :-1]) / (dz[..., 1:] + dz[..., :-1])
    zero = jnp.zeros_like(r[..., :1])
    grad = _cat(zero, grad_body)  # grad[j], j = 0..K-1 (grad[0] unused)
    grad_kp1 = _cat(grad[..., 1:], zero)  # grad[j+1]
    dz_km1 = _cat(zero, dz[..., :-1])  # dz[j-1]
    dz_kp1 = _cat(dz[..., 1:], zero)  # dz[j+1]

    if linear:
        interior = (grad_kp1 + grad) * dz  # F90 L533
    else:
        denom = dz_km1 + dz + dz_kp1
        interior = (
            grad_kp1 * (2.0 * dz_km1 + dz) + grad * (2.0 * dz_kp1 + dz)
        ) * dz / denom  # F90 L537-539

    top = 2.0 * grad[..., 1:2] * dz[..., 0:1]  # slope[0] (F90 L542)
    bot = 2.0 * grad[..., k - 1 : k] * dz[..., k - 1 : k]  # slope[K-1] (F90 L543)
    slope = _cat(top, interior[..., 1 : k - 1], bot)

    if limit:
        r_km1 = _cat(zero, r[..., :-1])
        r_kp1 = _cat(r[..., 1:], zero)
        r_min = jnp.minimum(jnp.minimum(r_km1, r), r_kp1)
        r_max = jnp.maximum(jnp.maximum(r_km1, r), r_kp1)
        limited = jnp.where(slope < 0.0, -1.0, 1.0) * jnp.minimum(
            jnp.minimum(jnp.abs(slope), 2.0 * (r - r_min)), 2.0 * (r_max - r)
        )
        # limiter applies to the interior only; end slopes -> 0 (F90 L560)
        slope = _cat(zero, limited[..., 1 : k - 1], zero)
    return slope


def _compute_weights(dz: Array) -> tuple[Array, Array, Array]:
    """4th-order interface weights ``zwt1, zwt2, zwt3`` — port of ``compute_weights``
    (F90 L572-634). Each is ``(..., K)``; only interior indices ``2..K-2`` are
    valid (used only there). ``zwt0`` (mask path) is not needed here.
    """
    zero = jnp.zeros_like(dz[..., :1])
    dz_km2 = _cat(zero, zero, dz[..., :-2])  # dz[k-2]
    dz_km1 = _cat(zero, dz[..., :-1])  # dz[k-1]
    dz_kp1 = _cat(dz[..., 1:], zero)  # dz[k+1]

    denom1 = 1.0 / (dz_km1 + dz)
    denom2 = 1.0 / (dz_km2 + dz_km1 + dz + dz_kp1)
    denom3 = 1.0 / (2.0 * dz_km1 + dz)
    denom4 = 1.0 / (dz_km1 + 2.0 * dz)
    num3 = dz_km2 + dz_km1
    num4 = dz + dz_kp1
    x = num3 * denom3 - num4 * denom4
    y = 2.0 * dz_km1 * dz
    zwt0 = dz_km1 * denom1
    zwt1 = zwt0 + x * y * denom1 * denom2
    zwt2 = dz_km1 * num3 * denom3 * denom2
    zwt3 = dz * num4 * denom4 * denom2
    return zwt1, zwt2, zwt3


def _interior_flux(scheme: int, w: Array, dz: Array, r: Array, dt: float) -> Array:
    """Interface fluxes for p = 1..K-1 (the interior interfaces), shape ``(..., K-1)``."""
    k = r.shape[-1]
    w_int = w[..., 1:k]  # w at interior interfaces p = 1..K-1
    r_lo = r[..., 0 : k - 1]  # r[p-1]
    r_hi = r[..., 1:k]  # r[p]

    if scheme == SECOND_CENTERED:
        return w_int * 0.5 * (r_hi + r_lo)

    if scheme == SECOND_CENTERED_WTS:
        wt = dz[..., 0 : k - 1] / (dz[..., 0 : k - 1] + dz[..., 1:k])
        return w_int * (r_lo + wt * (r_hi - r_lo))

    if scheme == FINITE_VOLUME_LINEAR:
        slp = _slope_z(r, dz)  # limit=True, linear=True (F90 L279)
        slp_lo = slp[..., 0 : k - 1]  # slp[p-1]
        slp_hi = slp[..., 1:k]  # slp[p]
        cn_pos = dt * w_int / dz[..., 0 : k - 1]  # w >= 0 (F90 L285)
        cn_neg = -dt * w_int / dz[..., 1:k]  # w <  0 (F90 L289)
        rst_pos = r_lo + 0.5 * slp_lo * (1.0 - cn_pos)
        rst_neg = r_hi - 0.5 * slp_hi * (1.0 - cn_neg)
        return w_int * jnp.where(w_int >= 0.0, rst_pos, rst_neg)

    if scheme in (FOURTH_CENTERED, FOURTH_CENTERED_WTS):
        # interior interfaces p = 2..K-2 use the 4th-order stencil; the two
        # innermost interfaces p = 1 and p = K-1 fall back to 2nd order.
        if scheme == FOURTH_CENTERED:
            c1, c2 = 7.0 / 12.0, 1.0 / 12.0
            r_p = r[..., 2 : k - 1]  # r[p],   p = 2..K-2
            r_pm1 = r[..., 1 : k - 2]  # r[p-1]
            r_pp1 = r[..., 3:k]  # r[p+1]
            r_pm2 = r[..., 0 : k - 3]  # r[p-2]
            rst_mid = c1 * (r_p + r_pm1) - c2 * (r_pp1 + r_pm2)  # F90 L260
            # 2nd-order at p = 1, p = K-1 (F90 L269-272)
            rst_p1 = 0.5 * (r[..., 1:2] + r[..., 0:1])
            rst_pkm1 = 0.5 * (r[..., k - 1 : k] + r[..., k - 2 : k - 1])
        else:  # FOURTH_CENTERED_WTS
            zwt1, zwt2, zwt3 = _compute_weights(dz)
            slp = _slope_z(r, dz, limit=False, linear=False)  # F90 L198
            r_p = r[..., 2 : k - 1]
            r_pm1 = r[..., 1 : k - 2]
            slp_p = slp[..., 2 : k - 1]
            slp_pm1 = slp[..., 1 : k - 2]
            rst_mid = (
                r_pm1
                + zwt1[..., 2 : k - 1] * (r_p - r_pm1)
                - zwt2[..., 2 : k - 1] * slp_p
                + zwt3[..., 2 : k - 1] * slp_pm1
            )  # F90 L219-220
            # 2nd-order-wts at p = 1, p = K-1 (F90 L229-234)
            wt0 = dz[..., 0:1] / (dz[..., 0:1] + dz[..., 1:2])
            rst_p1 = r[..., 0:1] + wt0 * (r[..., 1:2] - r[..., 0:1])
            wtn = dz[..., k - 2 : k - 1] / (dz[..., k - 2 : k - 1] + dz[..., k - 1 : k])
            rst_pkm1 = r[..., k - 2 : k - 1] + wtn * (r[..., k - 1 : k] - r[..., k - 2 : k - 1])
        rst = _cat(rst_p1, rst_mid, rst_pkm1)  # p = 1..K-1
        return w_int * rst

    if scheme == FINITE_VOLUME_PARABOLIC:
        return _ppm_interior_flux(w, dz, r, dt)

    raise NotImplementedError(
        f"vert_advection scheme {scheme} not ported "
        "(FINITE_VOLUME_PARABOLIC2 / Lin-2003 limiter is a follow-up)"
    )


def _ppm_reconstruct(r: Array, dz: Array) -> tuple[Array, Array]:
    """Piecewise-parabolic cell edge values with the Colella-Woodward limiter —
    port of the PPM reconstruction in ``vert_advection_3d`` (F90 L302-369).

    Returns ``(r_left, r_right)`` ``(..., K)``: the tracer values at each cell's
    upper and lower faces. Uses the unequal-spacing 4th-order interface weights
    (:func:`_compute_weights`) and the unlimited slope (:func:`_slope_z`,
    ``limit=False`` interior end handling but the PPM calls ``slope_z`` with its
    default ``limit=True``). Boundary cells fall back to the linear (slope-based)
    edge values (F90 L319-334).
    """
    k = r.shape[-1]
    slp = _slope_z(r, dz, limit=True, linear=False)  # F90 L303 slope_z(...,linear=.false.)
    zwt1, zwt2, zwt3 = _compute_weights(dz)

    # interior edge value r_left[c] for cells c = 2..K-2 (F90 L307-308):
    #   r[c-1] + zwt1[c]*(r[c]-r[c-1]) - zwt2[c]*slp[c] + zwt3[c]*slp[c-1]
    r_cm1 = _cat(jnp.zeros_like(r[..., :1]), r[..., :-1])
    slp_cm1 = _cat(jnp.zeros_like(slp[..., :1]), slp[..., :-1])
    rL_int = r_cm1 + zwt1 * (r - r_cm1) - zwt2 * slp + zwt3 * slp_cm1

    # assemble r_left by cell: linear edges at c = 0, 1, K-1 (F90 L319/331/333)
    r_left = _cat(
        r[..., 0:1] - 0.5 * slp[..., 0:1],  # c = 0 (ks)
        r[..., 1:2] - 0.5 * slp[..., 1:2],  # c = 1 (ks+1)
        rL_int[..., 2 : k - 1],  # c = 2..K-2
        r[..., k - 1 : k] - 0.5 * slp[..., k - 1 : k],  # c = K-1 (ke)
    )
    # r_right[c-1] = r_left[c] for c = 2..K-2 -> r_right[1..K-3]; linear elsewhere
    r_right = _cat(
        r[..., 0:1] + 0.5 * slp[..., 0:1],  # c = 0 (ks)
        r_left[..., 2 : k - 1],  # c = 1..K-3
        r[..., k - 2 : k - 1] + 0.5 * slp[..., k - 2 : k - 1],  # c = K-2 (ke-1)
        r[..., k - 1 : k] + 0.5 * slp[..., k - 1 : k],  # c = K-1 (ke)
    )

    # Colella-Woodward monotonicity limiter (F90 L351-368).
    test1 = (r_right - r) * (r - r_left) <= 0.0  # local extremum -> flatten (F90 L355)
    r_left = jnp.where(test1, r, r_left)
    r_right = jnp.where(test1, r, r_right)
    rm = r_right - r_left  # F90 L361
    a = rm * (r - 0.5 * (r_right + r_left))  # F90 L362
    b = rm * rm / 6.0  # F90 L363
    # overshoot corrections (F90 L364-365); a>b and a<-b are mutually exclusive.
    rL_ov = jnp.where(a > b, 3.0 * r - 2.0 * r_right, r_left)
    rR_ov = jnp.where(a < -b, 3.0 * r - 2.0 * r_left, r_right)
    interior = (jnp.arange(k) >= 1) & (jnp.arange(k) <= k - 2)  # F90 L360 skips ks/ke
    r_left = jnp.where(interior, rL_ov, r_left)
    r_right = jnp.where(interior, rR_ov, r_right)
    return r_left, r_right


def _ppm_interior_flux(w: Array, dz: Array, r: Array, dt: float) -> Array:
    """PPM interface fluxes for the interior interfaces ``p = 1..K-1`` (F90 L371-438).

    Reproduces the flux integral of the piecewise parabola over the departure
    region, **including the Courant>1 extension** (F90 L382-393 / L409-420). That
    extension is a departure-point integral: the walk that accumulates whole cells
    until the swept distance ``dt·|w|`` is covered is exactly a ``searchsorted`` on
    the cumulative-``dz`` prefix, so it needs no data-dependent loop. The single
    unified formula ``rst = (xx·rst_partial + rsum)/cn`` reduces to the plain
    single-cell PPM flux when ``cn <= 1`` (then ``rsum = 0``, ``xx = cn``, upwind
    cell = the adjacent cell).
    """
    k = r.shape[-1]
    r_left, r_right = _ppm_reconstruct(r, dz)
    tt = 2.0 / 3.0  # F90 L373

    zero_c = jnp.zeros_like(dz[..., :1])
    p_pref = _cat(zero_c, jnp.cumsum(dz, axis=-1))  # P[c] = sum_{c'<c} dz, (..., K+1)
    r_pref = _cat(jnp.zeros_like(r[..., :1]), jnp.cumsum(r, axis=-1))  # (..., K+1)

    w_int = w[..., 1:k]  # interfaces p = 1..K-1, (..., K-1)
    i_idx = jnp.arange(1, k)  # interface index i, broadcasts over batch
    p_i = p_pref[..., 1:k]  # P[i], (..., K-1)
    depart = p_i - dt * w_int  # departure position (same expression for both signs)

    def _gather(arr, idx):
        return jnp.take_along_axis(arr, jnp.broadcast_to(idx, depart.shape), axis=-1)

    def _branch(sign_pos):
        # upwind cell kk0 and the crossing cell kk_f (searchsorted on the prefix)
        if sign_pos:  # w >= 0: kk0 = i-1, walk up (decreasing c); stop where P[c] <= depart
            kk0 = i_idx - 1
            count = jnp.sum(p_pref[..., None, :] <= depart[..., :, None], axis=-1)
            kkf = jnp.clip(count - 1, 0, kk0)
        else:  # w < 0: kk0 = i, walk down (increasing c); stop where P[c] < depart
            kk0 = jnp.broadcast_to(i_idx, depart.shape)
            count = jnp.sum(p_pref[..., None, :] < depart[..., :, None], axis=-1)
            kkf = jnp.clip(count - 1, i_idx, k - 1)
        rLf = _gather(r_left, kkf)
        rRf = _gather(r_right, kkf)
        rf = _gather(r, kkf)
        dzf = _gather(dz, kkf)
        # Courant fraction in the partial cell kk_f, and rsum over the fully
        # crossed cells: the walk goes *up* (decreasing c) for w>=0 and *down*
        # (increasing c) for w<0, so the partial-cell offset and the swept range
        # mirror between branches.
        if sign_pos:
            xx = (_gather(p_pref, kkf + 1) - depart) / dzf  # (P[kk_f+1]-depart)/dz
            rsum = _gather(r_pref, i_idx) - _gather(r_pref, kkf + 1)
        else:
            xx = (depart - _gather(p_pref, kkf)) / dzf  # (depart-P[kk_f])/dz
            rsum = _gather(r_pref, kkf) - _gather(r_pref, i_idx)
        rm = rRf - rLf
        r6 = 6.0 * (rf - 0.5 * (rRf + rLf))
        # r6 -> 0 in the top (ks) / bottom (ke) cell (F90 L400 / L427)
        edge = (kkf == 0) if sign_pos else (kkf == k - 1)
        r6 = jnp.where(edge, 0.0, r6)
        if sign_pos:  # F90 L401
            rst_partial = rRf - 0.5 * xx * (rm - (1.0 - tt * xx) * r6)
        else:  # F90 L428
            rst_partial = rLf + 0.5 * xx * (rm + (1.0 - tt * xx) * r6)
        cn = jnp.abs(dt * w_int) / _gather(dz, kk0)  # F90 L379/L406 (dz of kk0)
        rst = (xx * rst_partial + rsum) / cn  # F90 L403/L430 (reduces to rst_partial if cn<=1)
        return w_int * rst

    return jnp.where(w_int >= 0.0, _branch(True), _branch(False))


def vert_advection(
    dt: float,
    w: Array,
    dz: Array,
    r: Array,
    scheme: int = VAN_LEER_LINEAR,
    form: int = FLUX_FORM,
    flags: int = 0,
    mask: Array | None = None,
) -> Array:
    """Vertical advective tendency — port of ``vert_advection_3d`` (F90 L70-478).

    ``w`` is ``(..., K+1)`` (interface velocities), ``dz`` and ``r`` are
    ``(..., K)``. Returns ``rdt`` ``(..., K)``. ``scheme`` / ``form`` / ``flags``
    are static integers (see the module-level constants); defaults match the
    Fortran (``VAN_LEER_LINEAR``, ``FLUX_FORM``). ``mask`` and the
    ``OUTFLOW_BOUNDARY`` flag are not supported (see module docstring).
    """
    if mask is not None:
        raise NotImplementedError("vert_advection mask (below-ground layers) not ported")
    if flags & OUTFLOW_BOUNDARY:
        raise NotImplementedError("vert_advection OUTFLOW_BOUNDARY not ported")
    do_weighted = bool(flags & WEIGHTED_TENDENCY)

    k = r.shape[-1]
    # boundary fluxes (non-outflow, F90 L166-167)
    flux_top = w[..., 0:1] * r[..., 0:1]
    flux_bot = w[..., k : k + 1] * r[..., k - 1 : k]
    flux = _cat(flux_top, _interior_flux(scheme, w, dz, r, dt), flux_bot)  # (..., K+1)

    d_flux = flux[..., 1 : k + 1] - flux[..., 0:k]  # flux[k+1] - flux[k]
    if form == FLUX_FORM:
        tendency = -d_flux  # F90 L450-459
    elif form == ADVECTIVE_FORM:
        d_w = w[..., 1 : k + 1] - w[..., 0:k]  # w[k+1] - w[k]
        tendency = -(d_flux - r * d_w)  # F90 L460-471
    else:
        raise ValueError(f"invalid vert_advection form {form}")

    return tendency if do_weighted else tendency / dz
