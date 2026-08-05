"""Generalized (hybrid) vertical-coordinate coefficients ``a`` (= pk) and ``b`` (= bk).

Faithful port of Isca's ``src/atmos_spectral/init/vert_coordinate.F90``
(``compute_vert_coord``). Given a number of levels and a coordinate option, it
returns the ``K + 1`` interface coefficients ``a`` and ``b`` such that the
interface pressures are

    p_half = a * reference_press + b * p_surf

(F90 L63). These feed ``press_and_geopot`` / ``implicit`` / ``spectral_dynamics``
as ``pk`` and ``bk`` — this is the vertical coordinate the whole column uses, and
a prerequisite of the ``spectral_dynamics`` assembly (queue item 5).

This is host-side initialization (run once), so it is plain NumPy — not on the
jit step path.

Options ported (F90 L135-154):

* ``'even_sigma'``   — uniform sigma (F90 L230); ``a = 0``, ``b`` linear in level.
* ``'uneven_sigma'`` — stretched sigma with surface resolution (F90 L248).
* ``'hybrid'``       — sigma near the surface blending to pure pressure aloft via
  a ``sin^2`` transition (F90 L141-147).
* ``'mcm'``          — hard-coded 14-level profile (F90 L296).
* ``'v197'``         — hard-coded 18-level profile (F90 L276).

The ``'input'`` option (F90 L135) reads ``pk``/``bk`` straight from a namelist —
pure I/O, not a computation; not ported here (a caller with explicit
coefficients passes them directly). Isca's default is ``'even_sigma'``.

Faithful subtlety (``hybrid``, F90 L142-146): the two ``compute_uneven_sigma``
calls share identical parameters, and the second call passes its arguments
**swapped** (``b_press, a_press``), so ``a_sigma = b_press = 0`` and
``a_press = b_sigma`` are the *same* stretched-sigma profile ``u``. The result
reduces to ``a = reference_press * u * (1 - f)`` and ``b = u * f`` with
``f = transition(u)``. Reproduced via the same two-call structure for clarity.

Fixtures: ``tests/fixtures/vert_coordinate_reference.npz`` from
``fortran_instrumentation/dump_vert_coordinate_reference.F90`` (real Fortran).
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def _compute_even_sigma(num_levels: int) -> tuple[Array, Array]:
    """``even_sigma`` — F90 L230-244."""
    a = np.zeros(num_levels + 1)
    b = np.zeros(num_levels + 1)
    b[:num_levels] = np.arange(num_levels) / float(num_levels)  # b(k)=(k-1)/N
    b[num_levels] = 1.0
    return a, b


def _compute_uneven_sigma(
    num_levels: int, scale_heights: float, surf_res: float, exponent: float, zero_top: bool
) -> tuple[Array, Array]:
    """``uneven_sigma`` — F90 L248-273."""
    a = np.zeros(num_levels + 1)
    b = np.zeros(num_levels + 1)
    s2 = 1.0 - surf_res
    k = np.arange(1, num_levels + 1)  # Fortran k = 1..num_levels
    zeta = 1.0 - (k - 1) / float(num_levels)
    z = surf_res * zeta + s2 * (zeta**exponent)
    b[:num_levels] = np.exp(-z * scale_heights)
    b[num_levels] = 1.0
    if zero_top:
        b[0] = 0.0
    return a, b


def _transition(p: Array, p_sigma: float, p_press: float) -> Array:
    """Smooth ``sin^2`` blend from pressure (0) to sigma (1) — F90 L161-183."""
    x = p - p_press
    xx = p_sigma - p_press
    t = np.sin(0.5 * np.pi * x / xx) ** 2
    t = np.where(p <= p_press, 0.0, t)
    t = np.where(p >= p_sigma, 1.0, t)
    return t


# hard-coded profiles (F90 L287-289, L306); bk has num_levels+1 entries.
_V197_BK = np.array(
    [0.0, .0089163, .0342936, .0740741, .1262002, .1886145, .2592592,
     .3360768, .4170096, .5000000, .5829904, .6639231, .7407407,
     .8113854, .8737997, .9259259, .9657064, .9910837, 1.0]
)
_MCM_BK = np.array(
    [0.0, .03, .0707, .1311, .2102, .3036, .4062, .5138, .6226, .7284,
     .8255, .9066, .9640, .9933, 1.0]
)


def compute_vert_coord(
    vert_coord_option: str,
    num_levels: int,
    scale_heights: float = 4.0,
    surf_res: float = 0.1,
    exponent: float = 2.5,
    p_press: float = 0.1,
    p_sigma: float = 0.3,
    reference_press: float = 101325.0,
) -> tuple[Array, Array]:
    """Vertical-coordinate coefficients ``(a, b)`` — port of ``compute_vert_coord`` (F90 L89-157).

    Returns ``a`` (= pk) and ``b`` (= bk), each length ``num_levels + 1``
    (interfaces). Defaults mirror Isca's ``spectral_dynamics_nml``. The ``'input'``
    (namelist) option is not ported (see module docstring).
    """
    if vert_coord_option in ("uneven_sigma", "hybrid"):
        if scale_heights == 0.0:
            raise ValueError("zero is an invalid value for scale_heights")
        if exponent == 0.0:
            raise ValueError("zero is an invalid value for exponent")
        if surf_res <= 0.0 or surf_res > 1.0:
            raise ValueError(f"surf_res must be in (0, 1], got {surf_res}")
    if vert_coord_option == "hybrid" and p_sigma < p_press:
        raise ValueError(f"p_sigma ({p_sigma}) must be >= p_press ({p_press})")

    if vert_coord_option == "even_sigma":
        return _compute_even_sigma(num_levels)
    if vert_coord_option == "uneven_sigma":
        return _compute_uneven_sigma(num_levels, scale_heights, surf_res, exponent, True)
    if vert_coord_option == "hybrid":
        a_sigma, b_sigma = _compute_uneven_sigma(
            num_levels, scale_heights, surf_res, exponent, False
        )
        # second call passes (b_press, a_press) swapped (F90 L143): a_press = the
        # stretched-sigma profile, b_press = 0.
        b_press, a_press = _compute_uneven_sigma(
            num_levels, scale_heights, surf_res, exponent, False
        )
        f = _transition(b_sigma, p_sigma, p_press)
        a = a_sigma * f + a_press * (1.0 - f)
        b = b_sigma * f + b_press * (1.0 - f)
        a = reference_press * a
        return a, b
    if vert_coord_option == "mcm":
        if num_levels != 14:
            raise ValueError(f"mcm requires num_levels == 14, got {num_levels}")
        return np.zeros(num_levels + 1), _MCM_BK.copy()
    if vert_coord_option == "v197":
        if num_levels != 18:
            raise ValueError(f"v197 requires num_levels == 18, got {num_levels}")
        return np.zeros(num_levels + 1), _V197_BK.copy()
    if vert_coord_option == "input":
        raise NotImplementedError(
            "vert_coord_option 'input' reads pk/bk from a namelist; pass explicit "
            "coefficients instead"
        )
    raise ValueError(f"'{vert_coord_option}' is not a valid vert_coord_option")
