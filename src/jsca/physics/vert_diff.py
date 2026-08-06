"""Implicit vertical diffusion — Isca's ``vert_diff.F90`` (gcm_vert_diff down/up).

Faithful port of the implicit (backward-Euler) tridiagonal vertical-diffusion
solver that applies the boundary-layer diffusivities (from
:mod:`jsca.physics.diffusivity`) to mix ``u``, ``v``, ``T`` and ``q`` up the
column each step, for the Frierson configuration
(``do_conserve_energy=.true.``, ``do_virtual=.false.``, single ``sphum`` tracer).

Because the surface fluxes of heat/moisture depend implicitly on the lowest-level
values, Isca splits the solve around the surface (mixed-layer) update:

* :func:`vert_diff_down` — the **down** pass: fully diffuses momentum (with the
  surface stress as the implicit bottom boundary condition and the frictional
  dissipative heating fed into ``dt_t``), and runs the forward elimination for
  ``T``/``q`` — storing the tridiagonal coefficients (``e``, ``f_t``, ``f_q``)
  and the surface-coupling terms in a :class:`TriSurf`.
* the mixed layer (item 8) then updates ``TriSurf.delta_t`` from the surface
  energy balance;
* :func:`vert_diff_up` — the **up** pass: back-substitutes ``T``/``q``.

The momentum surface stress is applied here (``diff_surface``); the T/q surface
coupling is left to the mixed layer, so calling down→up directly (as the fixture
does) is a well-defined diffusion with the T/q surface flux held at its stored
value.

**Numerics (F90 compute_e/compute_f/vert_diff_up).** For a field ``xi`` the
implicit step ``(1 - delt·D)·xi^{n+1} = xi^n + explicit`` is a tridiagonal system
with ``a = -mu·nu(k+1)·delt`` (super-diagonal), ``c = -mu·nu(k)·delt`` (sub),
``b = 1 - a - c``. Thomas elimination gives ``e``/``g`` (forward) and ``f`` (from
the RHS); the back-substitution ``dt_xi(k) = e(k)·dt_xi(k+1) + f(k)`` completes
it. ``mu = g/Δp`` (layer mass), ``nu = ρ_half·K/Δz`` (diffusive conductance).

**Scope:** the Frierson path (no ``kbot``/eta topography, no MCM pressure-level
form, ``do_virtual=.false.``). No documented deviation — pure arithmetic, so
fixtures hit machine precision.

Layout: column physics, **level axis last**; ``u``/``v``/``t``/``q``/``diff_*``/
``p_full``/``z_full`` are ``(..., K)``; ``p_half`` is ``(..., K+1)``; the surface
fluxes/derivatives are ``(...)``. ``k=0`` top … ``K-1`` surface.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jsca import constants

Array = jnp.ndarray
_GCP = constants.GRAV / constants.CP_AIR


class TriSurf(NamedTuple):
    """Tridiagonal state carried from the down pass to the up pass (a pytree).

    ``delta_t``/``delta_q`` are the surface-level RHS terms the mixed layer may
    update; ``e``/``f_t``/``f_q`` are the elimination coefficients for the
    back-substitution. ``mu_delt_n``/``nu_n``/``e_n1``/``dflux`` expose the
    surface coupling the mixed-layer solve needs (item 8).
    """

    e: Array          # (..., K) elimination coefficient (heat/moisture)
    f_t: Array        # (..., K)
    f_q: Array        # (..., K)
    delta_t: Array    # (...,)
    delta_q: Array    # (...,)
    mu_delt_n: Array  # (...,) surface layer mass * delt
    nu_n: Array       # (...,) surface conductance
    e_n1: Array       # (...,) e at level K-2
    dflux: Array      # (...,) -nu_n*(1-e_n1)


# --- tridiagonal building blocks (level axis last) ------------------------
def _compute_mu(p_half):
    return constants.GRAV / (p_half[..., 1:] - p_half[..., :-1])


def _compute_nu(diff, p_half, t, z_full):
    """Diffusive conductance at interior half levels (F90 ``compute_nu``,
    do_virtual=.false.). ``nu[..., 0] = 0`` (top, unused)."""
    rho_half = 2.0 * p_half[..., 1:-1] / (constants.RDGAS * (t[..., 1:] + t[..., :-1]))
    nu_int = rho_half * diff[..., 1:] / (z_full[..., :-1] - z_full[..., 1:])
    return jnp.concatenate([jnp.zeros_like(nu_int[..., :1]), nu_int], axis=-1)


def _explicit_tend(mu, nu, xi, dt_xi):
    """Add the explicit diffusive flux divergence (F90 ``explicit_tend``)."""
    flux_int = nu[..., 1:] * (xi[..., 1:] - xi[..., :-1])
    fluxx = jnp.concatenate([jnp.zeros_like(flux_int[..., :1]), flux_int], axis=-1)
    d_int = mu[..., :-1] * (fluxx[..., 1:] - fluxx[..., :-1])
    d_bot = -mu[..., -1:] * fluxx[..., -1:]
    return dt_xi + jnp.concatenate([d_int, d_bot], axis=-1)


def _compute_abc(delt, mu, nu):
    a = jnp.concatenate([-mu[..., :-1] * nu[..., 1:] * delt,
                         jnp.zeros_like(mu[..., :1])], axis=-1)
    c = jnp.concatenate([jnp.zeros_like(mu[..., :1]),
                         -mu[..., 1:] * nu[..., 1:] * delt], axis=-1)
    b = 1.0 - a - c
    return a, b, c


def _compute_e(a, b, c):
    """Forward elimination coefficients ``e``, ``g`` (F90 ``compute_e``)."""
    a_f = jnp.moveaxis(a, -1, 0)
    b_f = jnp.moveaxis(b, -1, 0)
    c_f = jnp.moveaxis(c, -1, 0)
    e0 = -a_f[0] / b_f[0]

    def step(e_prev, kv):
        ak, bk, ck = kv
        gk = 1.0 / (bk + ck * e_prev)
        ek = -ak * gk
        return ek, (ek, gk)

    _, (e_rest, g_rest) = jax.lax.scan(step, e0, (a_f[1:], b_f[1:], c_f[1:]))
    e = jnp.concatenate([e0[None], e_rest], axis=0)
    g = jnp.concatenate([jnp.zeros_like(e0)[None], g_rest], axis=0)
    return jnp.moveaxis(e, 0, -1), jnp.moveaxis(g, 0, -1)


def _compute_f(dt_xi, b, c, g):
    """RHS elimination coefficient ``f`` (F90 ``compute_f``)."""
    d_f = jnp.moveaxis(dt_xi, -1, 0)
    b_f = jnp.moveaxis(b, -1, 0)
    c_f = jnp.moveaxis(c, -1, 0)
    g_f = jnp.moveaxis(g, -1, 0)
    f0 = d_f[0] / b_f[0]

    def step(f_prev, kv):
        dk, ck, gk = kv
        fk = (dk - ck * f_prev) * gk
        return fk, fk

    _, f_rest = jax.lax.scan(step, f0, (d_f[1:], c_f[1:], g_f[1:]))
    return jnp.moveaxis(jnp.concatenate([f0[None], f_rest], axis=0), 0, -1)


def _vert_diff_up(e, f, delta_n, delt):
    """Back-substitution (F90 ``vert_diff_up``). Returns ``dt_xi`` ``(..., K)``."""
    e_f = jnp.moveaxis(e, -1, 0)
    f_f = jnp.moveaxis(f, -1, 0)
    bottom = delta_n / delt

    def step(carry, kv):
        ek, fk = kv
        new = ek * carry + fk
        return new, new

    _, out = jax.lax.scan(step, bottom, (e_f[:-1], f_f[:-1]), reverse=True)
    return jnp.moveaxis(jnp.concatenate([out, bottom[None]], axis=0), 0, -1)


def _diff_surface(mu_delt, nu_n, e_n1, f_delt_n1, dflux_datmos, flux, factor, delta_xi):
    """Fold the surface flux into the surface-level RHS (F90 ``diff_surface``)."""
    fff = 1.0 / factor
    dflux = -nu_n * (1.0 - e_n1)
    delta = delta_xi + mu_delt * nu_n * f_delt_n1
    delta = (delta + mu_delt * flux * fff) / (1.0 - mu_delt * (dflux + dflux_datmos * fff))
    return delta, flux + dflux_datmos * delta


def _surface_terms(mu, nu, e, f, dt_xi1, delt):
    """Extract the lowest-level tridiagonal terms (no kbot: kb = K-1)."""
    mu_delt_n = mu[..., -1] * delt
    nu_n = nu[..., -1]
    e_n1 = e[..., -2]
    f_delt_n1 = f[..., -1 - 1] * delt   # f at level K-2
    delta_n = dt_xi1[..., -1] * delt
    return mu_delt_n, nu_n, e_n1, f_delt_n1, delta_n


def vert_diff_down(delt, u, v, t, q, diff_m, diff_t, p_half, p_full, z_full,
                   tau_u, tau_v, dtau_du, dtau_dv,
                   dt_u=0.0, dt_v=0.0, dt_t=0.0, dt_q=0.0):
    """Down pass: momentum solve + T/q forward elimination.

    Returns ``(dt_u, dt_v, diss_heat, tri)`` — the final momentum tendencies, the
    frictional heating, and a :class:`TriSurf` for the up pass. Input tendencies
    default to zero.
    """
    dt_u = jnp.broadcast_to(jnp.asarray(dt_u, u.dtype), u.shape)
    dt_v = jnp.broadcast_to(jnp.asarray(dt_v, u.dtype), u.shape)
    dt_t = jnp.broadcast_to(jnp.asarray(dt_t, u.dtype), u.shape)
    dt_q = jnp.broadcast_to(jnp.asarray(dt_q, u.dtype), u.shape)

    tt = t + z_full * _GCP
    mu = _compute_mu(p_half)

    # --- momentum (uv_vert_diff), with diff_m ---
    nu_m = _compute_nu(diff_m, p_half, t, z_full)
    a, b, c = _compute_abc(delt, mu, nu_m)
    e_m, g_m = _compute_e(a, b, c)
    dt_u1 = _explicit_tend(mu, nu_m, u, dt_u)
    dt_v1 = _explicit_tend(mu, nu_m, v, dt_v)
    f_u = _compute_f(dt_u1, b, c, g_m)
    f_v = _compute_f(dt_v1, b, c, g_m)
    mu_delt_n, nu_n_m, e_n1_m, f_u_dn1, delta_u_n = _surface_terms(mu, nu_m, e_m, f_u, dt_u1, delt)
    _, _, _, f_v_dn1, delta_v_n = _surface_terms(mu, nu_m, e_m, f_v, dt_v1, delt)
    delta_u_n, tau_u = _diff_surface(
        mu_delt_n, nu_n_m, e_n1_m, f_u_dn1, dtau_du, tau_u, 1.0, delta_u_n)
    delta_v_n, tau_v = _diff_surface(
        mu_delt_n, nu_n_m, e_n1_m, f_v_dn1, dtau_dv, tau_v, 1.0, delta_v_n)
    dt_u_out = _vert_diff_up(e_m, f_u, delta_u_n, delt)
    dt_v_out = _vert_diff_up(e_m, f_v, delta_v_n, delt)

    # frictional dissipative heating (do_conserve_energy) -> into dt_t
    du = dt_u_out - dt_u
    dv = dt_v_out - dt_v
    diss_heat = -(1.0 / constants.CP_AIR) * (
        (u + 0.5 * delt * du) * du + (v + 0.5 * delt * dv) * dv)
    dt_t = dt_t + diss_heat

    # --- heat / moisture forward elimination, with diff_t ---
    nu_t = _compute_nu(diff_t, p_half, t, z_full)
    a2, b2, c2 = _compute_abc(delt, mu, nu_t)
    e_g, g_t = _compute_e(a2, b2, c2)
    dt_t1 = _explicit_tend(mu, nu_t, tt, dt_t)
    dt_q1 = _explicit_tend(mu, nu_t, q, dt_q)
    f_t = _compute_f(dt_t1, b2, c2, g_t)
    f_q = _compute_f(dt_q1, b2, c2, g_t)
    mu_delt_n, nu_n, e_n1, f_t_dn1, delta_t_n = _surface_terms(mu, nu_t, e_g, f_t, dt_t1, delt)
    _, _, _, f_q_dn1, delta_q_n = _surface_terms(mu, nu_t, e_g, f_q, dt_q1, delt)

    delta_t = delta_t_n + mu_delt_n * nu_n * f_t_dn1
    delta_q = delta_q_n + mu_delt_n * nu_n * f_q_dn1
    tri = TriSurf(e=e_g, f_t=f_t, f_q=f_q, delta_t=delta_t, delta_q=delta_q,
                  mu_delt_n=mu_delt_n, nu_n=nu_n, e_n1=e_n1,
                  dflux=-nu_n * (1.0 - e_n1))
    return dt_u_out, dt_v_out, diss_heat, tri


def vert_diff_up(delt, tri: TriSurf):
    """Up pass: back-substitute ``T`` and ``q``. Returns ``(dt_t, dt_q)``."""
    dt_t = _vert_diff_up(tri.e, tri.f_t, tri.delta_t, delt)
    dt_q = _vert_diff_up(tri.e, tri.f_q, tri.delta_q, delt)
    return dt_t, dt_q
