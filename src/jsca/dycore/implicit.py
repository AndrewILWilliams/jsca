"""Semi-implicit gravity-wave correction for the spectral dynamics.

Faithful port of Isca's ``src/atmos_spectral/model/implicit.F90``. The
semi-implicit scheme treats the linear gravity-wave terms implicitly: each step
the divergence tendency is solved through a per-total-wavenumber vertical matrix
``(I + xi^2 l(l+1)/a^2 · div_mat)^{-1}`` (``xi = alpha·dt``), and the temperature
and log-surface-pressure tendencies are corrected consistently.

Layout: spectral storage ``(m, n)`` with the level axis **last**; complex
spectral fields are ``(m, n, k)`` and time-level stacks ``(m, n, k, t)`` /
``(m, n, t)``, exactly the Fortran ``(0:,0:,:,:)`` order (`k = 0` top …
`k = K-1` surface). Reference-state vectors are real, length ``K`` (full) or
``K+1`` (half). ``previous``/``current`` are 0-based static time indices.

Structure (mirrors the Fortran):

* :func:`implicit_init` — builds the dt-independent reference state: the
  linearised reference ``ln p`` (via the already-ported
  :func:`jsca.dycore.pressure_variables`), the ``d ln p / d ps`` derivatives
  (``del_ln_p_half`` analytic, ``del_ln_p_full`` by the Fortran's ±eps finite
  difference, ``implicit.F90`` L144-160), and the divergence coupling matrix
  ``div_mat`` and pressure-gradient vector ``h`` from :func:`_build_matrix`
  (L171-217). Returns an :class:`ImplicitParams` pytree.
* :func:`build_wave_matrices` — the dt-dependent inverses
  ``(I + factor·div_mat)^{-1}``, ``factor = xi^2 l(l+1)/a^2`` (L221-237), stacked
  over total wavenumber ``l = 0 … num_total_wavenumbers``. Uses the ported
  :func:`jsca.dycore.invert` (LAPACK; documented deviation, see
  ``matrix_invert.py``).
* :func:`implicit_correction` — one correction (L241-286): ``adjust_dt_divs``
  (L289-325), the per-wavenumber vertical solve, then the consistent T / ln ps
  updates.

The private linear operators — :func:`_linear_tp_tendency` (L414-503),
:func:`_linear_geopotential` (L329-359) and :func:`_pres_grad_funct` (L389-411)
— are written once over ``(..., K)`` arrays so the same code serves the real 1-D
build-time calls and the complex 3-D step-time calls (the Fortran's ``_1d``
variants just wrap the ``_3d`` ones). Sequential vertical sums are reproduced by
prefix sums with the Fortran association (shifted ``cumsum``, not
``cumsum - x``), so agreement is limited only by the LAPACK inversion.

Fortran state fragility (noted, not reproduced): ``implicit_end`` leaves the
module's cached ``dt`` set and a fresh ``implicit_init`` allocates ``wave_matrix``
without filling it, so a re-init followed by a correction at the *same* ``dt``
would read an uninitialised matrix (build only fires on ``dt`` change, L260-264).
Here the wave matrices are an explicit return value keyed to ``dt`` — no hidden
cache, no stale-state hazard.

Only ``implicit_init``/``implicit_correction``/``implicit_end`` are public in the
Fortran, so the fixtures validate this whole file end-to-end through
:func:`implicit_correction`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from jsca import constants
from jsca.dycore.matrix_invert import invert
from jsca.dycore.press_and_geopot import pressure_variables

Array = jnp.ndarray

_EPS = 1.0e-5  # finite-difference step for del_ln_p_full (implicit.F90 L94)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ImplicitParams:
    """dt-independent reference state for the semi-implicit correction."""

    pk: Array  # (K+1,)
    bk: Array  # (K+1,)
    dpk: Array  # (K,)  pk[k+1]-pk[k]
    dbk: Array  # (K,)  bk[k+1]-bk[k]
    ref_t: Array  # (K,)   reference temperature profile
    ref_ln_p_half: Array  # (K+1,)
    ref_ln_p_full: Array  # (K,)
    del_ln_p_half: Array  # (K+1,)  d ln p_half / d ps
    del_ln_p_full: Array  # (K,)    d ln p_full / d ps
    div_mat: Array  # (K, K)  divergence coupling matrix
    h: Array  # (K,)    pressure-gradient reference vector
    eigen: Array  # (m, n)  l(l+1)/a^2
    wavenumber: Array  # (m, n)  total wavenumber l (int)
    ref_surf_p: float
    alpha: float
    num_total_wavenumbers: int
    vert_difference_option: str
    radius: float
    rdgas: float
    cp_air: float

    def tree_flatten(self):
        children = (
            self.pk, self.bk, self.dpk, self.dbk, self.ref_t,
            self.ref_ln_p_half, self.ref_ln_p_full,
            self.del_ln_p_half, self.del_ln_p_full,
            self.div_mat, self.h, self.eigen, self.wavenumber,
        )
        aux = (
            self.ref_surf_p, self.alpha, self.num_total_wavenumbers,
            self.vert_difference_option, self.radius, self.rdgas, self.cp_air,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)


# --------------------------- linear operators --------------------------------


def _linear_tp_tendency(
    div: Array,
    ref_t: Array,
    ref_ln_p_half: Array,
    ref_ln_p_full: Array,
    ref_surf_p: float,
    pk: Array,
    bk: Array,
    dpk: Array,
    dbk: Array,
    vert_difference_option: str,
    rdgas: float,
    cp_air: float,
) -> tuple[Array, Array]:
    """Linearised (T, ln ps) tendency from divergence — port of ``linear_tp_tendency_3d``.

    ``div`` is ``(..., K)``; returns ``(dt_p_surf (...,), dt_t (..., K))``.
    """
    kappa = rdgas / cp_air
    dp = dpk + dbk * ref_surf_p  # (K,)
    dp_inv = 1.0 / dp
    dmean = div * dp  # (..., K)
    dmean_incl = jnp.cumsum(dmean, axis=-1)  # running total including level k
    # exclusive prefix = the Fortran dmean_tot *before* adding level k; use a
    # shift (not dmean_incl - dmean) to keep the exact accumulation association.
    dmean_excl = jnp.concatenate(
        [jnp.zeros_like(dmean[..., :1]), dmean_incl[..., :-1]], axis=-1
    )
    dmean_tot = dmean_incl[..., -1:]  # (..., 1) grand total

    if vert_difference_option == "simmons_and_burridge":
        dlog_1 = ref_ln_p_half[1:] - ref_ln_p_full  # (K,)
        dlog_3 = ref_ln_p_half[1:] - ref_ln_p_half[:-1]  # (K,)
        dt_t = -kappa * ref_t * (dmean_excl * dlog_3 + dmean * dlog_1) * dp_inv
    elif vert_difference_option == "mcm":
        p_full_ref = 0.5 * (pk[1:] + pk[:-1]) + 0.5 * (bk[1:] + bk[:-1]) * ref_surf_p
        dt_t = -(kappa * ref_t / p_full_ref) * (dmean_excl + 0.5 * dmean)
    else:
        raise ValueError(f"invalid vert_difference_option {vert_difference_option!r}")

    dt_p_surf = -dmean_incl[..., -1]  # (...,)

    # vertical velocity at half levels: vert_vel[..., k] for k=1..K = -dmean_incl[k-1];
    # index 0 is unused. Then interior levels get the +dmean_tot*bk(k) correction.
    vert_vel = jnp.concatenate(
        [jnp.zeros_like(dmean[..., :1]), -dmean_incl], axis=-1
    )  # (..., K+1)
    interior = vert_vel[..., 1:-1] + dmean_tot * bk[1:-1]  # levels 1..K-1
    vert_vel = jnp.concatenate(
        [vert_vel[..., :1], interior, vert_vel[..., -1:]], axis=-1
    )

    # temp[..., k] = -vert_vel[k]*(t[k]-t[k-1]) for k=1..K-1; temp[0]=temp[K]=0.
    temp_interior = -vert_vel[..., 1:-1] * (ref_t[1:] - ref_t[:-1])  # (..., K-1)
    zero = jnp.zeros_like(dmean[..., :1])
    temp = jnp.concatenate([zero, temp_interior, zero], axis=-1)  # (..., K+1)

    dt_t = dt_t + 0.5 * dp_inv * (temp[..., 1:] + temp[..., :-1])
    return dt_p_surf, dt_t


def _linear_geopotential(
    del_t: Array,
    del_ln_p_half: Array,
    del_ln_p_full: Array,
    ref_t: Array,
    ref_ln_p_half: Array,
    ref_ln_p_full: Array,
    rdgas: float,
) -> Array:
    """Linearised geopotential — port of ``linear_geopotential_3d``.

    ``del_t``/``del_ln_p_full`` are ``(..., K)``, ``del_ln_p_half`` is ``(..., K+1)``;
    returns ``geopot (..., K)``. Integrated downward from a zero top half level.
    """
    inc = rdgas * (
        del_t * (ref_ln_p_half[1:] - ref_ln_p_half[:-1])
        + ref_t * (del_ln_p_half[..., 1:] - del_ln_p_half[..., :-1])
    )  # (..., K); index 0 unused downstream
    # geopot_half[k] = sum_{j=k}^{K-1} inc[j], accumulated top-down (matches Fortran).
    gh_body = jnp.flip(jnp.cumsum(jnp.flip(inc, axis=-1), axis=-1), axis=-1)
    geopot_half = jnp.concatenate(
        [gh_body, jnp.zeros_like(inc[..., :1])], axis=-1
    )  # (..., K+1); [K]=0
    geopot = geopot_half[..., 1:] + rdgas * (
        del_t * (ref_ln_p_half[1:] - ref_ln_p_full)
        + ref_t * (del_ln_p_half[..., 1:] - del_ln_p_full)
    )
    return geopot


def _pres_grad_funct(
    ref_t: Array,
    ref_ln_p_half: Array,
    ref_ln_p_full: Array,
    ref_surf_p: float,
    bk: Array,
    dpk: Array,
    dbk: Array,
    vert_difference_option: str,
    rdgas: float,
) -> Array:
    """Reference pressure-gradient term — port of ``pres_grad_funct`` (1-D)."""
    if vert_difference_option == "simmons_and_burridge":
        dlog_1 = ref_ln_p_half[1:] - ref_ln_p_full
        dlog_2 = ref_ln_p_full - ref_ln_p_half[:-1]
        return rdgas * ref_t * (bk[1:] * dlog_1 + bk[:-1] * dlog_2) / (dpk + dbk * ref_surf_p)
    elif vert_difference_option == "mcm":
        return rdgas * ref_t / ref_surf_p
    raise ValueError(f"invalid vert_difference_option {vert_difference_option!r}")


# ------------------------------ init / build ---------------------------------


def _build_matrix(p: ImplicitParams) -> tuple[Array, Array]:
    """Build ``div_mat`` and ``h`` — port of ``build_matrix`` (implicit.F90 L171-217)."""
    k = p.ref_t.shape[0]
    ident = jnp.eye(k)

    # unit divergence in each level -> nu, tau (linear_tp_tendency of e_k)
    dt_p_surf, dt_t = _linear_tp_tendency(
        ident, p.ref_t, p.ref_ln_p_half, p.ref_ln_p_full, p.ref_surf_p,
        p.pk, p.bk, p.dpk, p.dbk, p.vert_difference_option, p.rdgas, p.cp_air,
    )
    nu = -dt_p_surf  # (K,)
    tau = -dt_t.T  # tau[level, input]

    # unit temperature in each level -> gamma (linear_geopotential of e_k)
    zero_half = jnp.zeros((k, k + 1))
    zero_full = jnp.zeros((k, k))
    gamma = _linear_geopotential(
        ident, zero_half, zero_full, p.ref_t, p.ref_ln_p_half, p.ref_ln_p_full, p.rdgas
    ).T  # gamma[level, input]

    h1 = _pres_grad_funct(
        p.ref_t, p.ref_ln_p_half, p.ref_ln_p_full, p.ref_surf_p,
        p.bk, p.dpk, p.dbk, p.vert_difference_option, p.rdgas,
    )
    h2 = _linear_geopotential(
        jnp.zeros(k), p.del_ln_p_half, p.del_ln_p_full,
        p.ref_t, p.ref_ln_p_half, p.ref_ln_p_full, p.rdgas,
    )
    h = h1 + h2

    div_mat = jnp.outer(h, nu) + gamma @ tau
    return div_mat, h


def implicit_init(
    pk: Array,
    bk: Array,
    ref_temperature: Array,
    ref_surf_p: float,
    num_total_wavenumbers: int,
    eigen: Array,
    wavenumber: Array,
    alpha: float,
    vert_difference_option: str = "simmons_and_burridge",
    radius: float = constants.RADIUS,
    rdgas: float = constants.RDGAS,
    cp_air: float = constants.CP_AIR,
) -> ImplicitParams:
    """Build the reference-state matrices — port of ``implicit_init`` (L79-167).

    ``pk``/``bk`` are the ``K+1`` hybrid coefficients, ``ref_temperature`` the
    ``K``-level reference profile, ``eigen``/``wavenumber`` the ``l(l+1)/a^2`` and
    ``l`` tables (positive convention, as :func:`jsca.dycore.spectral_damping`).
    """
    pk = jnp.asarray(pk)
    bk = jnp.asarray(bk)
    ref_t = jnp.asarray(ref_temperature)
    dpk = pk[1:] - pk[:-1]
    dbk = bk[1:] - bk[:-1]

    ps = jnp.asarray(float(ref_surf_p))
    _, ref_ln_p_half, _, ref_ln_p_full = pressure_variables(pk, bk, ps, vert_difference_option)

    # del_ln_p_half = d ln p_half / d ps, analytic (L144-152)
    body = bk[1:] / (pk[1:] + bk[1:] * ref_surf_p)
    if bool(pk[0] == 0.0):
        head = jnp.asarray(1.0 / ref_surf_p)
    else:
        head = bk[0] / (pk[0] + bk[0] * ref_surf_p)
    del_ln_p_half = jnp.concatenate([head.reshape(1), body])

    # del_ln_p_full = d ln p_full / d ps by centred finite difference (L154-160)
    _, _, _, ln_p_full_lo = pressure_variables(
        pk, bk, jnp.asarray(ref_surf_p * (1.0 - 0.5 * _EPS)), vert_difference_option
    )
    _, _, _, ln_p_full_hi = pressure_variables(
        pk, bk, jnp.asarray(ref_surf_p * (1.0 + 0.5 * _EPS)), vert_difference_option
    )
    del_ln_p_full = (ln_p_full_hi - ln_p_full_lo) / (_EPS * ref_surf_p)

    p = ImplicitParams(
        pk=pk, bk=bk, dpk=dpk, dbk=dbk, ref_t=ref_t,
        ref_ln_p_half=ref_ln_p_half, ref_ln_p_full=ref_ln_p_full,
        del_ln_p_half=del_ln_p_half, del_ln_p_full=del_ln_p_full,
        div_mat=jnp.zeros((ref_t.shape[0], ref_t.shape[0])),
        h=jnp.zeros(ref_t.shape[0]),
        eigen=jnp.asarray(eigen), wavenumber=jnp.asarray(wavenumber),
        ref_surf_p=float(ref_surf_p), alpha=float(alpha),
        num_total_wavenumbers=int(num_total_wavenumbers),
        vert_difference_option=vert_difference_option,
        radius=float(radius), rdgas=float(rdgas), cp_air=float(cp_air),
    )
    div_mat, h = _build_matrix(p)
    return replace(p, div_mat=div_mat, h=h)


def build_wave_matrices(p: ImplicitParams, dt: float) -> Array:
    """Inverse vertical matrices per total wavenumber — port of ``build_wave_matrices``.

    Returns ``(num_total_wavenumbers+1, K, K)``; index ``l`` is
    ``(I + xi^2 l(l+1)/a^2 · div_mat)^{-1}`` with ``xi = alpha·dt``.
    """
    xi = dt * p.alpha
    k = p.ref_t.shape[0]
    ls = jnp.arange(p.num_total_wavenumbers + 1)
    factor = xi * xi * ls * (ls + 1.0) / p.radius**2  # (L+1,)
    mats = jnp.eye(k)[None, :, :] + factor[:, None, None] * p.div_mat[None, :, :]
    inv, _ = invert(mats)
    return inv


def _ltt(p: ImplicitParams, div: Array) -> tuple[Array, Array]:
    return _linear_tp_tendency(
        div, p.ref_t, p.ref_ln_p_half, p.ref_ln_p_full, p.ref_surf_p,
        p.pk, p.bk, p.dpk, p.dbk, p.vert_difference_option, p.rdgas, p.cp_air,
    )


def _lgeo(p: ImplicitParams, del_t: Array, del_ln_p_half: Array, del_ln_p_full: Array) -> Array:
    return _linear_geopotential(
        del_t, del_ln_p_half, del_ln_p_full,
        p.ref_t, p.ref_ln_p_half, p.ref_ln_p_full, p.rdgas,
    )


def _adjust_dt_divs(
    p: ImplicitParams,
    dt_divs: Array,
    dt_ts: Array,
    dt_ln_ps: Array,
    divs: Array,
    ts: Array,
    ln_ps: Array,
    xi: float,
    previous: int,
    current: int,
) -> tuple[Array, Array, Array]:
    """Port of ``adjust_dt_divs`` (implicit.F90 L289-325)."""
    divs_temp = divs[..., previous] - divs[..., current]
    dt_ps_temp, dt_ts_temp = _ltt(p, divs_temp)
    dt_ts = dt_ts + dt_ts_temp
    dt_ln_ps = dt_ln_ps + dt_ps_temp / p.ref_surf_p

    ts_temp = ts[..., previous] - ts[..., current] + xi * dt_ts
    ps_temp = ln_ps[..., previous] - ln_ps[..., current] + xi * dt_ln_ps

    zero_half = jnp.zeros_like(ts_temp[..., :1]).repeat(ts_temp.shape[-1] + 1, axis=-1)
    zero_full = jnp.zeros_like(ts_temp)
    geopot = _lgeo(p, ts_temp, zero_half, zero_full)

    dt_divs = dt_divs + p.eigen[..., None] * (
        geopot + p.h * (ps_temp[..., None] * p.ref_surf_p)
    )
    return dt_divs, dt_ts, dt_ln_ps


def implicit_correction(
    p: ImplicitParams,
    wave_matrix: Array,
    dt: float,
    dt_divs: Array,
    dt_ts: Array,
    dt_ln_ps: Array,
    divs: Array,
    ts: Array,
    ln_ps: Array,
    previous: int,
    current: int,
) -> tuple[Array, Array, Array]:
    """One semi-implicit correction — port of ``implicit_correction`` (L241-286).

    ``dt_divs``/``dt_ts`` are ``(m, n, K)`` tendencies, ``dt_ln_ps`` is ``(m, n)``;
    ``divs``/``ts`` are ``(m, n, K, t)`` and ``ln_ps`` is ``(m, n, t)`` time-level
    stacks. ``wave_matrix`` comes from :func:`build_wave_matrices` at this ``dt``.
    Returns the updated ``(dt_divs, dt_ts, dt_ln_ps)``.
    """
    xi = dt * p.alpha

    dt_divs, dt_ts, dt_ln_ps = _adjust_dt_divs(
        p, dt_divs, dt_ts, dt_ln_ps, divs, ts, ln_ps, xi, previous, current
    )

    # per-(m, n) vertical solve: dt_divs <- wave_matrix[l] @ dt_divs, only where
    # l = wavenumber(m, n) <= num_total_wavenumbers.
    lidx = p.wavenumber
    valid = lidx <= p.num_total_wavenumbers
    lclip = jnp.where(valid, lidx, 0)
    w = wave_matrix[lclip]  # (m, n, K, K)
    solved = jnp.einsum("mnij,mnj->mni", w, dt_divs)
    dt_divs = jnp.where(valid[..., None], solved, dt_divs)

    dt_ps_temp, dt_ts_temp = _ltt(p, dt_divs)
    dt_ts = dt_ts + xi * dt_ts_temp
    dt_ln_ps = dt_ln_ps + xi * dt_ps_temp / p.ref_surf_p
    return dt_divs, dt_ts, dt_ln_ps
