"""Pressures and geopotential heights on the hybrid (pk, bk) vertical grid.

Faithful port of Isca's ``src/atmos_spectral/model/press_and_geopot.F90``
("computes pressure and geopotential by integrating the hydrostatic/ideal
gas equations exactly").

Conventions:

* Column functions put the **level axis last**: ``ps`` has shape ``(...,)``,
  full-level fields ``(..., K)``, half-level fields ``(..., K + 1)``, with
  k = 0 the model top and k = K the surface — exactly the Fortran index
  order, so Fortran fixtures compare without transposition.
* ``pk``, ``bk`` are the K+1 hybrid coefficients (Pa): ``p_half = pk + bk * ps``.
  Isca's ``uneven_sigma`` option has ``pk = 0`` everywhere.
* ``vert_difference_option`` is ``'simmons_and_burridge'`` (Isca's default)
  or ``'mcm'``; a static Python string, resolved at trace time.
* The Fortran's module state (``press_and_geopot_init`` storing pk/bk/flags)
  becomes explicit arguments, per the jsca design rules.
* ``ln_top_level_factor = -1.0`` is hard-coded as in the Fortran
  (``ln_p_full(1) = ln_p_half(2) - 1`` for a p_top = 0 grid).

Summation order in the hydrostatic integration reproduces the Fortran's
bottom-up sequential loop exactly (cumsum seeded with the surface term), so
agreement with fixtures is limited only by libm/fma differences.
"""

from __future__ import annotations

import jax.numpy as jnp

from jsca import constants

Array = jnp.ndarray

_LN_TOP_LEVEL_FACTOR = -1.0


def half_level_pressures(pk: Array, bk: Array, surface_p: Array) -> Array:
    """``p_half[..., k] = pk[k] + bk[k] * ps`` — port of ``half_level_pressures``."""
    return pk + bk * surface_p[..., None]


def pressure_variables(
    pk: Array,
    bk: Array,
    surface_p: Array,
    vert_difference_option: str = "simmons_and_burridge",
) -> tuple[Array, Array, Array, Array]:
    """Port of ``pressure_variables``: ``(p_half, ln_p_half, p_full, ln_p_full)``.

    Simmons & Burridge (1981) full-level placement:
        alpha_k     = 1 - p_half_k (ln p_half_{k+1} - ln p_half_k) / (p_half_{k+1} - p_half_k)
        ln p_full_k = ln p_half_{k+1} - alpha_k
    with the special top treatment when p_half_top = 0 (pk[0] = bk[0] = 0):
    ``ln_p_half[0] = 0`` (unused sentinel, as in Fortran) and
    ``ln_p_full[0] = ln_p_half[1] + ln_top_level_factor``.
    """
    p_half = half_level_pressures(pk, bk, surface_p)
    top_is_zero = bool(pk[0] == 0.0) and bool(bk[0] == 0.0)

    if vert_difference_option == "simmons_and_burridge":
        if top_is_zero:
            ln_p_half_body = jnp.log(p_half[..., 1:])
            ln_p_half = jnp.concatenate(
                [jnp.zeros_like(p_half[..., :1]), ln_p_half_body], axis=-1
            )
            alpha = 1.0 - p_half[..., 1:-1] * (
                ln_p_half[..., 2:] - ln_p_half[..., 1:-1]
            ) / (p_half[..., 2:] - p_half[..., 1:-1])
            ln_p_full_body = ln_p_half[..., 2:] - alpha
            ln_p_full = jnp.concatenate(
                [ln_p_half[..., 1:2] + _LN_TOP_LEVEL_FACTOR, ln_p_full_body], axis=-1
            )
        else:
            ln_p_half = jnp.log(p_half)
            alpha = 1.0 - p_half[..., :-1] * (
                ln_p_half[..., 1:] - ln_p_half[..., :-1]
            ) / (p_half[..., 1:] - p_half[..., :-1])
            ln_p_full = ln_p_half[..., 1:] - alpha
        p_full = jnp.exp(ln_p_full)
    elif vert_difference_option == "mcm":
        p_full = 0.5 * (p_half[..., 1:] + p_half[..., :-1])
        ln_p_full = jnp.log(p_full)
        if top_is_zero:
            ln_p_half = jnp.concatenate(
                [jnp.zeros_like(p_half[..., :1]), jnp.log(p_half[..., 1:])], axis=-1
            )
        else:
            ln_p_half = jnp.log(p_half)
    else:
        raise ValueError(
            f"{vert_difference_option!r} is not a valid vert_difference_option"
        )
    return p_half, ln_p_half, p_full, ln_p_full


def _virtual_temperature(
    t: Array, q: Array | None, use_virtual_temperature: bool, rdgas: float, rvgas: float
) -> Array:
    if use_virtual_temperature:
        if q is None:
            raise ValueError("q must be provided when use_virtual_temperature=True")
        return t * (1.0 + (rvgas / rdgas - 1.0) * q)
    return t


def compute_geopotential(
    pk: Array,
    t_grid: Array,
    ln_p_half: Array,
    ln_p_full: Array,
    surf_geopotential: Array,
    q_grid: Array | None = None,
    use_virtual_temperature: bool = False,
    rdgas: float = constants.RDGAS,
    rvgas: float = constants.RVGAS,
) -> tuple[Array, Array]:
    """Port of ``compute_geopotential`` → ``(geopot_full, geopot_half)`` [J/kg].

    Hydrostatic integration upward from ``geopot_half[..., K] = Phi_s``:
        Phi_half[k] = Phi_half[k+1] + Rd Tv[k] (ln p_half[k+1] - ln p_half[k])
        Phi_full[k] = Phi_half[k+1] + Rd Tv[k] (ln p_half[k+1] - ln p_full[k])
    with ``Phi_half[0] = 0`` when the top half-level pressure is zero (pk[0]=0).
    """
    virtual_t = _virtual_temperature(t_grid, q_grid, use_virtual_temperature, rdgas, rvgas)
    increments = rdgas * virtual_t * (ln_p_half[..., 1:] - ln_p_half[..., :-1])
    # Fortran accumulates bottom-up starting from Phi_s:
    #   half[K-1] = Phi_s + inc[K-1]; half[K-2] = half[K-1] + inc[K-2]; ...
    # Reproduce the exact association: cumsum over [Phi_s, inc_K-1, inc_K-2, ...].
    seq = jnp.concatenate(
        [surf_geopotential[..., None], jnp.flip(increments, axis=-1)], axis=-1
    )
    geopot_half_body = jnp.flip(jnp.cumsum(seq, axis=-1)[..., 1:], axis=-1)
    geopot_half = jnp.concatenate(
        [geopot_half_body, surf_geopotential[..., None]], axis=-1
    )
    if bool(pk[0] == 0.0):
        # Fortran leaves geopot_half(:, :, 1) = 0 (loop stops at ktop = 2)
        geopot_half = geopot_half.at[..., 0].set(0.0)
    geopot_full = geopot_half[..., 1:] + rdgas * virtual_t * (
        ln_p_half[..., 1:] - ln_p_full
    )
    return geopot_full, geopot_half


def compute_z_bot(
    pk: Array,
    bk: Array,
    psg: Array,
    tg: Array,
    qg: Array | None = None,
    use_virtual_temperature: bool = False,
    vert_difference_option: str = "simmons_and_burridge",
    rdgas: float = constants.RDGAS,
    rvgas: float = constants.RVGAS,
    grav: float = constants.GRAV,
) -> Array:
    """Port of ``compute_z_bot``: height [m] of the lowest full level above the surface."""
    p_half_bot = pk[-1] + psg * bk[-1]
    p_half_nxt = pk[-2] + psg * bk[-2]
    ln_p_half_bot = jnp.log(p_half_bot)
    if vert_difference_option == "simmons_and_burridge":
        ln_p_half_nxt = jnp.log(p_half_nxt)
        alpha = 1.0 - p_half_nxt * (ln_p_half_bot - ln_p_half_nxt) / (
            p_half_bot - p_half_nxt
        )
        ln_p_full_bot = ln_p_half_bot - alpha
    elif vert_difference_option == "mcm":
        ln_p_full_bot = jnp.log(0.5 * (p_half_bot + p_half_nxt))
    else:
        raise ValueError(f"invalid vert_difference_option {vert_difference_option!r}")
    virtual_t = _virtual_temperature(tg, qg, use_virtual_temperature, rdgas, rvgas)
    return rdgas * virtual_t * (ln_p_half_bot - ln_p_full_bot) / grav


def compute_pressures_and_heights(
    pk: Array,
    bk: Array,
    t_grid: Array,
    ps_grid: Array,
    surf_geopotential: Array,
    q_grid: Array | None = None,
    use_virtual_temperature: bool = False,
    vert_difference_option: str = "simmons_and_burridge",
    rdgas: float = constants.RDGAS,
    rvgas: float = constants.RVGAS,
    grav: float = constants.GRAV,
) -> tuple[Array, Array, Array, Array]:
    """Port of ``compute_pressures_and_heights`` → ``(z_full, z_half, p_full, p_half)`` [m, Pa]."""
    p_half, ln_p_half, p_full, ln_p_full = pressure_variables(
        pk, bk, ps_grid, vert_difference_option
    )
    geopot_full, geopot_half = compute_geopotential(
        pk,
        t_grid,
        ln_p_half,
        ln_p_full,
        surf_geopotential,
        q_grid,
        use_virtual_temperature,
        rdgas,
        rvgas,
    )
    return geopot_full / grav, geopot_half / grav, p_full, p_half
