"""Monin-Obukhov surface-layer similarity — Isca's ``monin_obukhov_kernel.F90``.

Faithful port of the ``_PURE`` Monin-Obukhov kernel that Isca's surface-flux
scheme rests on, for the default ``monin_obukhov_nml`` the Frierson aquaplanet
uses: ``stable_option = 1``, ``rich_crit = 2.0``, ``zeta_trans = 0.5``,
``neutral = .false.``, ``drag_min = 1e-5`` (solver ``error = 1e-4``,
``zeta_min = 1e-6``, ``small = 1e-4``, ``max_iter = 20``).

Monin-Obukhov similarity relates the surface turbulent exchange to the near-
surface stratification through a stability parameter ``zeta = z/L`` (``L`` the
Obukhov length). ``mo_drag`` (F90 ``monin_obukhov_drag_1d``) returns the bulk
drag coefficients ``cd_m``/``cd_t``/``cd_q`` and the friction/buoyancy scales
``u_star``/``b_star`` given the bulk Richardson number; ``mo_profile`` (F90
``monin_obukhov_profile_1d``) returns the interpolation ratios ``del_m``/
``del_t``/``del_q`` from the lowest level to reference heights (the 10 m / 2 m
diagnostics).

**The solver (F90 ``monin_obukhov_solve_zeta``).** For a non-neutral column the
integral similarity functions ``f_m``/``f_t``/``f_q`` (the effective log-law
denominators) depend on ``zeta``, which itself depends on them — so Isca solves
for ``zeta`` by Newton iteration from the bulk Richardson number, up to
``max_iter`` steps with a per-point convergence test. Here that is a
``lax.fori_loop`` carrying ``(zeta, done)``; converged points freeze, and the
final ``f_*`` are re-evaluated from the converged ``zeta`` (identical, since
``f_*`` are functions of ``zeta`` alone, and ``zeta -> 0`` reproduces the
``|zeta| < zeta_min`` short-circuit ``f_m = ln(z/z0)`` etc.).

Scope: ``stable_option = 1`` (the Frierson default). The unstable branch is
option-independent; the ``stable_option = 2`` stable branch is not ported (out of
scope for the aquaplanet, flagged here). No documented deviation — pure
arithmetic (``log``/``atan``/powers), so fixtures hit the log/exp tolerance.

Layout: surface-layer quantities are 1-D over horizontal points (``(N,)``);
everything is vectorized, the solver included.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jsca import constants

Array = jnp.ndarray


@dataclass(frozen=True)
class MOParams:
    """Static ``monin_obukhov_nml`` configuration (hashable; static jit arg)."""

    vonkarm: float = constants.VONKARM
    grav: float = constants.GRAV
    rich_crit: float = 2.0
    zeta_trans: float = 0.5      # stable_option=2 only (not exercised)
    drag_min: float = 1.0e-5
    neutral: bool = False
    stable_option: int = 1
    error: float = 1.0e-4
    zeta_min: float = 1.0e-6
    small: float = 1.0e-4
    max_iter: int = 20


# ---- stability similarity functions (stable_option = 1) -------------------
def _phi_m(zeta, rich_crit):
    """Differential similarity function for momentum (F90 ``derivative_m``)."""
    b_stab = 1.0 / rich_crit
    zu = jnp.minimum(zeta, 0.0)
    phi_u = jnp.sqrt((1.0 - 16.0 * zu) ** (-0.5))       # (1-16 zeta)^(-1/4)
    zs = jnp.maximum(zeta, 0.0)
    phi_s = 1.0 + zs * (5.0 + b_stab * zs) / (1.0 + zs)
    return jnp.where(zeta < 0.0, phi_u, phi_s)


def _phi_t(zeta, rich_crit):
    """Differential similarity function for heat/tracers (F90 ``derivative_t``)."""
    b_stab = 1.0 / rich_crit
    zu = jnp.minimum(zeta, 0.0)
    phi_u = (1.0 - 16.0 * zu) ** (-0.5)
    zs = jnp.maximum(zeta, 0.0)
    phi_s = 1.0 + zs * (5.0 + b_stab * zs) / (1.0 + zs)
    return jnp.where(zeta < 0.0, phi_u, phi_s)


def _psi_m(zeta, zeta_0, ln_z_z0, rich_crit):
    """Integral similarity function for momentum (F90 ``integral_m``)."""
    b_stab = 1.0 / rich_crit
    zu = jnp.minimum(zeta, 0.0)
    zu0 = jnp.minimum(zeta_0, 0.0)
    x = jnp.sqrt(jnp.sqrt(1.0 - 16.0 * zu))
    x_0 = jnp.sqrt(jnp.sqrt(1.0 - 16.0 * zu0))
    x1 = 1.0 + x
    x1_0 = 1.0 + x_0
    num = x1 * x1 * (1.0 + x * x)
    denom = x1_0 * x1_0 * (1.0 + x_0 * x_0)
    psi_u = ln_z_z0 - jnp.log(num / denom) + 2.0 * (jnp.arctan(x) - jnp.arctan(x_0))
    zs = jnp.maximum(zeta, 0.0)
    zs0 = jnp.maximum(zeta_0, 0.0)
    psi_s = ln_z_z0 + (5.0 - b_stab) * jnp.log((1.0 + zs) / (1.0 + zs0)) \
        + b_stab * (zs - zs0)
    return jnp.where(zeta < 0.0, psi_u, psi_s)


def _psi_tq(zeta, zeta_x, ln_z_zx, rich_crit):
    """Integral similarity function for heat OR moisture (F90 ``integral_tq``).

    Called once per scalar; ``zeta_x``/``ln_z_zx`` are the heat or moisture
    roughness quantities.
    """
    b_stab = 1.0 / rich_crit
    zu = jnp.minimum(zeta, 0.0)
    zux = jnp.minimum(zeta_x, 0.0)
    x = jnp.sqrt(1.0 - 16.0 * zu)
    x_x = jnp.sqrt(1.0 - 16.0 * zux)
    psi_u = ln_z_zx - 2.0 * jnp.log((1.0 + x) / (1.0 + x_x))
    zs = jnp.maximum(zeta, 0.0)
    zsx = jnp.maximum(zeta_x, 0.0)
    psi_s = ln_z_zx + (5.0 - b_stab) * jnp.log((1.0 + zs) / (1.0 + zsx)) \
        + b_stab * (zs - zsx)
    return jnp.where(zeta < 0.0, psi_u, psi_s)


def _solve_zeta(p: MOParams, rich, z, z0, zt, zq):
    """Newton solve for ``zeta`` → ``(f_m, f_t, f_q)`` (F90 ``solve_zeta``).

    All arrays ``(N,)``. Points with ``|zeta| < zeta_min`` short-circuit to the
    neutral log ratios; the rest iterate.
    """
    z_z0 = z / z0
    z_zt = z / zt
    z_zq = z / zq
    ln_z_z0 = jnp.log(z_z0)
    ln_z_zt = jnp.log(z_zt)
    ln_z_zq = jnp.log(z_zq)

    # initial guess (F90 L349-356)
    zeta0 = rich * ln_z_z0 * ln_z_z0 / ln_z_zt
    zeta0 = jnp.where(rich >= 0.0, zeta0 / (1.0 - rich / p.rich_crit), zeta0)

    def body(_i, carry):
        zeta, done = carry
        newly_small = (~done) & (jnp.abs(zeta) < p.zeta_min)
        zeta = jnp.where(newly_small, 0.0, zeta)
        done = done | newly_small

        zsafe = jnp.where(zeta == 0.0, 1.0, zeta)
        rzeta = 1.0 / zsafe
        zeta_0 = zeta / z_z0
        zeta_t = zeta / z_zt      # f_q (zeta_q) is not needed for the Newton step

        phi_m = _phi_m(zeta, p.rich_crit)
        phi_m_0 = _phi_m(zeta_0, p.rich_crit)
        phi_t = _phi_t(zeta, p.rich_crit)
        phi_t_0 = _phi_t(zeta_t, p.rich_crit)
        f_m = _psi_m(zeta, zeta_0, ln_z_z0, p.rich_crit)
        f_t = _psi_tq(zeta, zeta_t, ln_z_zt, p.rich_crit)

        df_m = (phi_m - phi_m_0) * rzeta
        df_t = (phi_t - phi_t_0) * rzeta
        rich_1 = zeta * f_t / (f_m * f_m)
        d_rich = rich_1 * (rzeta + df_t / f_t - 2.0 * df_m / f_m)
        correction = (rich - rich_1) / d_rich
        corr = jnp.minimum(jnp.abs(correction), jnp.abs(correction / zsafe))

        active = ~done
        still = active & (corr > p.error)
        zeta = jnp.where(still, zeta + correction, zeta)
        done = done | (active & (corr <= p.error))
        return zeta, done

    zeta, _ = jax.lax.fori_loop(0, p.max_iter, body, (zeta0, jnp.zeros_like(rich, bool)))

    # f_* from the converged zeta (zeta==0 reproduces the neutral short-circuit)
    zeta_0 = zeta / z_z0
    zeta_t = zeta / z_zt
    zeta_q = zeta / z_zq
    f_m = _psi_m(zeta, zeta_0, ln_z_z0, p.rich_crit)
    f_t = _psi_tq(zeta, zeta_t, ln_z_zt, p.rich_crit)
    f_q = _psi_tq(zeta, zeta_q, ln_z_zq, p.rich_crit)
    return f_m, f_t, f_q


def mo_drag(p: MOParams, pt, pt0, z, z0, zt, zq, speed):
    """Bulk drag coefficients and friction scales (F90 ``monin_obukhov_drag_1d``).

    Args are 1-D ``(N,)``: potential temperatures ``pt`` (air) / ``pt0``
    (surface), height ``z``, roughness lengths ``z0``/``zt``/``zq``, wind
    ``speed``. Returns ``(drag_m, drag_t, drag_q, u_star, b_star)`` (``stable_option=1``,
    non-neutral).
    """
    r_crit = 0.95 * p.rich_crit
    sqrt_drag_min = jnp.sqrt(p.drag_min)

    delta_b = p.grav * (pt0 - pt) / pt0
    rich = -z * delta_b / (speed * speed + p.small)
    zz = jnp.maximum(jnp.maximum(z, z0), jnp.maximum(zt, zq))

    f_m, f_t, f_q = _solve_zeta(p, rich, zz, z0, zt, zq)
    us = jnp.maximum(p.vonkarm / f_m, sqrt_drag_min)
    bs = jnp.maximum(p.vonkarm / f_t, sqrt_drag_min)
    qs = jnp.maximum(p.vonkarm / f_q, sqrt_drag_min)

    # rich >= r_crit: collapse to drag_min (F90 L213-222)
    crit = rich >= r_crit
    drag_m = jnp.where(crit, p.drag_min, us * us)
    drag_t = jnp.where(crit, p.drag_min, us * bs)
    drag_q = jnp.where(crit, p.drag_min, us * qs)
    us_out = jnp.where(crit, sqrt_drag_min, us)
    bs_out = jnp.where(crit, sqrt_drag_min, bs)
    u_star = us_out * speed
    b_star = bs_out * delta_b
    return drag_m, drag_t, drag_q, u_star, b_star


def mo_profile(p: MOParams, zref, zref_t, z, z0, zt, zq, u_star, b_star):
    """Reference-height interpolation ratios (F90 ``monin_obukhov_profile_1d``).

    Returns ``(del_m, del_t, del_q)`` — the fractional profile factors from the
    lowest level to ``zref`` (momentum) / ``zref_t`` (heat, moisture). Non-neutral,
    ``stable_option=1``.
    """
    ln_z_z0 = jnp.log(z / z0)
    ln_z_zt = jnp.log(z / zt)
    ln_z_zq = jnp.log(z / zq)
    ln_z_zref = jnp.log(z / zref)
    ln_z_zref_t = jnp.log(z / zref_t)

    has_ustar = u_star > 0.0
    mo_length_inv = jnp.where(has_ustar, -p.vonkarm * b_star / (u_star * u_star), 0.0)
    zeta = z * mo_length_inv
    zeta_0 = z0 * mo_length_inv
    zeta_t = zt * mo_length_inv
    zeta_q = zq * mo_length_inv
    zeta_ref = zref * mo_length_inv
    zeta_ref_t = zref_t * mo_length_inv

    f_m = _psi_m(zeta, zeta_0, ln_z_z0, p.rich_crit)
    f_m_ref = _psi_m(zeta, zeta_ref, ln_z_zref, p.rich_crit)
    f_t = _psi_tq(zeta, zeta_t, ln_z_zt, p.rich_crit)
    f_q = _psi_tq(zeta, zeta_q, ln_z_zq, p.rich_crit)
    f_t_ref = _psi_tq(zeta, zeta_ref_t, ln_z_zref_t, p.rich_crit)
    f_q_ref = f_t_ref   # F90 passes zeta_ref_t/ln_z_zref_t for both t and q ref

    del_m = 1.0 - f_m_ref / f_m
    del_t = 1.0 - f_t_ref / f_t
    del_q = 1.0 - f_q_ref / f_q
    return del_m, del_t, del_q
