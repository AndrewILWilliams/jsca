"""Spectral (hyper)diffusion damping and polar / eddy sponges.

Faithful port of Isca's ``src/atmos_spectral/model/spectral_damping.F90``.
Damping is applied *implicitly* to spectral tendencies each step:

    dt_spec <- (dt_spec - D * spec) / (1 + D * dt)

where ``D`` (``damping_eff`` in the Fortran) is a per-(m, n) coefficient built
once at init from the Laplacian eigenvalues ``eigen(m, n) = l(l+1)/a^2``,
``l = m + n`` (Isca's ``get_eigen_laplacian``; see ``spherical.F90`` lines
180-192). Note that Isca's ``get_eigen_laplacian`` returns the **positive**
``l(l+1)/a^2``; jsca's :func:`jsca.grid.spectral.laplacian_eigenvalues` returns
its negative (the actual Laplacian eigenvalue ``-l(l+1)/a^2``), so pass
``-laplacian_eigenvalues(...)`` here. Storage is GFDL ``(m, n)`` with the level
axis **last** for the 3-D fields, matching the Fortran ``(m, n, k)`` and index
``k = 1`` (top) mapping to Python index ``0``.

Init (``spectral_damping_init``, F90 L56-168) supports three
``damping_option`` values (F90 L126-155):

* ``'resolution_dependent'``   : ``coeff * (eigen / eigen(0, num_spherical-1))**order``
* ``'resolution_independent'`` : ``coeff * eigen**order``
* ``'exponential_cutoff'``     : K.S. Smith et al. (JFM 2002) filter on
  ``sqrt(eigen)`` above a cutoff wavenumber (F90 L131-147); zero below cutoff.

with independent (coeff, order) for vorticity and divergence (defaulting to the
base values, F90 L97-119), an optional linear drag ``+ damping_coeff_r`` on the
generic field (F90 L157-159), and three sponges (F90 L161-163):

* ``eddy_sponge``   ``= eddy_sponge_coeff * eigen``            (m /= 0)
* ``zmu_sponge(n)`` ``= zmu_sponge_coeff  * eigen(0, n)``      (zonal-mean u)
* ``zmv_sponge(n)`` ``= zmv_sponge_coeff  * eigen(0, n)``      (zonal-mean v)

The sponges act only on the **first vertical level** of vorticity / divergence
(F90 L231-243 / L276-288): the eddy sponge for ``m /= 0`` and the zonal-mean
sponge for ``m == 0``. :func:`spectral_damping_init` precombines them into
``sponge_vor`` / ``sponge_div`` (eddy array with its ``m = 0`` row replaced by
``zmu`` / ``zmv``) so the compute path is a single uniform expression.

Fortran subtleties preserved / noted
-------------------------------------
* ``damping_option_exponential`` is *module state* in the Fortran (L45) and
  ``spectral_damping_end`` does **not** reset it (L320-328); re-initialising with
  a non-exponential option after an exponential one therefore leaves the sticky
  flag set. Harmless in practice (init happens once per run) but a latent bug.
  Here the flag is an explicit per-instance field, so re-init is always clean.
* The generic 2-D and 3-D ``compute`` routines are one function; the level loop
  in the Fortran (L195-197) is pure broadcasting over the trailing axis.
* Exponential ``damping_eff`` (F90 L188/L219/L264/L307) is
  ``(exp(log(dt*coeff + 1) * damping) - 1) / dt`` with the *base/vor/div* local
  coefficients respectively.

All compute functions are pure and jit/vmap/scan-safe: :class:`SpectralDamping`
is a registered pytree whose numeric tables are dynamic children and whose
``exponential`` flag + local coefficients are static aux data, so the Python
branch on ``exponential`` resolves at trace time.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

Array = jnp.ndarray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SpectralDamping:
    """Precomputed damping tables (the port of the module's private state)."""

    damping: Array  # generic field, (m, n)
    damping_vor: Array  # vorticity, (m, n)
    damping_div: Array  # divergence, (m, n)
    eddy_sponge: Array  # eddy_sponge_coeff * eigen, (m, n)
    zmu_sponge: Array  # zmu_sponge_coeff * eigen(0, :), (n,)
    zmv_sponge: Array  # zmv_sponge_coeff * eigen(0, :), (n,)
    sponge_vor: Array  # eddy_sponge with row m=0 replaced by zmu_sponge, (m, n)
    sponge_div: Array  # eddy_sponge with row m=0 replaced by zmv_sponge, (m, n)
    exponential: bool  # static: exponential_cutoff option selected
    coeff_local: float  # static: base damping_coeff (exponential_cutoff only)
    coeff_vor_local: float  # static: vorticity damping_coeff
    coeff_div_local: float  # static: divergence damping_coeff

    def tree_flatten(self):
        children = (
            self.damping,
            self.damping_vor,
            self.damping_div,
            self.eddy_sponge,
            self.zmu_sponge,
            self.zmv_sponge,
            self.sponge_vor,
            self.sponge_div,
        )
        aux = (self.exponential, self.coeff_local, self.coeff_vor_local, self.coeff_div_local)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children, *aux)


def spectral_damping_init(
    eigen: Array,
    damping_coeff: float,
    damping_order: int,
    damping_option: str,
    cutoff_wn: int = 0,
    eddy_sponge_coeff: float = 0.0,
    zmu_sponge_coeff: float = 0.0,
    zmv_sponge_coeff: float = 0.0,
    damping_coeff_vor: float | None = None,
    damping_order_vor: int | None = None,
    damping_coeff_div: float | None = None,
    damping_order_div: int | None = None,
    damping_coeff_r: float | None = None,
) -> SpectralDamping:
    """Build the damping tables — port of ``spectral_damping_init`` (F90 L56-168).

    ``eigen`` is Isca's positive ``get_eigen_laplacian`` output, shape
    ``(num_fourier+1, num_spherical+1)``; ``num_spherical`` is inferred as
    ``eigen.shape[1] - 1``. The resolution-dependent / exponential branches
    reference ``eigen(0, num_spherical-1)`` — the largest resolved total
    wavenumber on the ``m = 0`` column — exactly as the Fortran.
    """
    eigen = jnp.asarray(eigen)
    num_spherical = eigen.shape[1] - 1
    sqrt_eigen = jnp.sqrt(eigen)

    coeff_vor = damping_coeff if damping_coeff_vor is None else damping_coeff_vor
    coeff_div = damping_coeff if damping_coeff_div is None else damping_coeff_div
    order_vor = damping_order if damping_order_vor is None else damping_order_vor
    order_div = damping_order if damping_order_div is None else damping_order_div

    exponential = False
    if damping_option == "resolution_dependent":
        ref = eigen[0, num_spherical - 1]
        damping = damping_coeff * (eigen / ref) ** damping_order
        damping_vor = coeff_vor * (eigen / ref) ** order_vor
        damping_div = coeff_div * (eigen / ref) ** order_div
    elif damping_option == "exponential_cutoff":
        # K.S. Smith et al. JFM 2002 filter (F90 L131-147). damping_order_local
        # == the base damping_order (F90 L123); vor/div are set equal to the
        # generic field before any linear drag.
        exponential = True
        se_cut = sqrt_eigen[0, cutoff_wn]
        se_max = sqrt_eigen[0, num_spherical - 1]
        base = ((sqrt_eigen - se_cut) / (se_max - se_cut)) ** damping_order
        mask = eigen / eigen[0, cutoff_wn] > 1.0
        damping = jnp.where(mask, base, 0.0)
        damping_vor = damping
        damping_div = damping
    elif damping_option == "resolution_independent":
        damping = damping_coeff * eigen**damping_order
        damping_vor = coeff_vor * eigen**order_vor
        damping_div = coeff_div * eigen**order_div
    else:
        raise ValueError(
            f'"{damping_option}" is an invalid value for damping_option'
        )

    if damping_coeff_r is not None:  # linear drag on the generic field (F90 L157-159)
        damping = damping + damping_coeff_r

    zmu_sponge = zmu_sponge_coeff * eigen[0, :]
    zmv_sponge = zmv_sponge_coeff * eigen[0, :]
    eddy_sponge = eddy_sponge_coeff * eigen
    sponge_vor = eddy_sponge.at[0, :].set(zmu_sponge)
    sponge_div = eddy_sponge.at[0, :].set(zmv_sponge)

    return SpectralDamping(
        damping=damping,
        damping_vor=damping_vor,
        damping_div=damping_div,
        eddy_sponge=eddy_sponge,
        zmu_sponge=zmu_sponge,
        zmv_sponge=zmv_sponge,
        sponge_vor=sponge_vor,
        sponge_div=sponge_div,
        exponential=exponential,
        coeff_local=float(damping_coeff),
        coeff_vor_local=float(coeff_vor),
        coeff_div_local=float(coeff_div),
    )


def _damping_eff(damping: Array, exponential: bool, coeff: float, current_dt: float) -> Array:
    """``damping_eff`` — plain table, or the exponential transform (F90 L187-191)."""
    if exponential:
        return (jnp.exp(jnp.log(current_dt * coeff + 1.0) * damping) - 1.0) / current_dt
    return damping


def _apply_bulk(damping_eff: Array, spec: Array, dt_spec: Array, current_dt: float) -> Array:
    """Implicit damping ``coeff*(dt_spec - D*spec)`` with ``coeff = 1/(1+D*dt)``.

    ``damping_eff`` is ``(m, n)``; ``spec``/``dt_spec`` are ``(m, n)`` or
    ``(m, n, k)``. The Fortran level loop is broadcasting over the trailing axis.
    """
    d = damping_eff.reshape(damping_eff.shape + (1,) * (dt_spec.ndim - damping_eff.ndim))
    coeff = 1.0 / (1.0 + d * current_dt)
    return coeff * (dt_spec - d * spec)


def compute_spectral_damping(
    params: SpectralDamping, spec: Array, dt_spec: Array, current_dt: float
) -> Array:
    """Generic implicit damping — port of ``compute_spectral_damping_2d/3d``.

    Returns the updated ``dt_spec`` (the Fortran mutates it in place). Works for
    both 2-D ``(m, n)`` and 3-D ``(m, n, k)`` spectral fields.
    """
    d = _damping_eff(params.damping, params.exponential, params.coeff_local, current_dt)
    return _apply_bulk(d, spec, dt_spec, current_dt)


def compute_spectral_damping_vor(
    params: SpectralDamping, vor: Array, dt_vor: Array, current_dt: float
) -> Array:
    """Vorticity damping + top-level sponge — port of ``compute_spectral_damping_vor``.

    ``vor``/``dt_vor`` are ``(m, n, k)``. After the bulk damping the eddy sponge
    (``m /= 0``) and zonal-mean-u sponge (``m == 0``) are applied to the first
    level only (F90 L231-243); ``sponge_vor`` carries the combined coefficient.
    """
    d = _damping_eff(params.damping_vor, params.exponential, params.coeff_vor_local, current_dt)
    dt_vor = _apply_bulk(d, vor, dt_vor, current_dt)
    return _apply_top_sponge(params.sponge_vor, vor, dt_vor, current_dt)


def compute_spectral_damping_div(
    params: SpectralDamping, div: Array, dt_div: Array, current_dt: float
) -> Array:
    """Divergence damping + top-level sponge — port of ``compute_spectral_damping_div``.

    As :func:`compute_spectral_damping_vor` but with the zonal-mean-v sponge on
    the ``m == 0`` row (F90 L276-288).
    """
    d = _damping_eff(params.damping_div, params.exponential, params.coeff_div_local, current_dt)
    dt_div = _apply_bulk(d, div, dt_div, current_dt)
    return _apply_top_sponge(params.sponge_div, div, dt_div, current_dt)


def _apply_top_sponge(sponge: Array, field: Array, dt_field: Array, current_dt: float) -> Array:
    """``(dt - s*field)/(1 + s*dt)`` on the first level only; ``s`` is ``(m, n)``."""
    top = (dt_field[..., 0] - sponge * field[..., 0]) / (1.0 + sponge * current_dt)
    return dt_field.at[..., 0].set(top)
