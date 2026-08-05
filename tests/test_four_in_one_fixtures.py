"""Tier-1 tests for four_in_one against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_four_in_one_reference.F90``.
``four_in_one`` is a *private* routine of the (un-stubbable) spectral_dynamics
module, so its body is compiled VERBATIM from the pinned source into a thin
scaffold module (``four_in_one_wrapper.F90``) that only supplies the
module-variable environment it references — no numerics are reimplemented.
Regeneration recipe in that file's header.

Fortran storage is ``(lon, lat, level)`` with level LAST — exactly the port's
``(..., K)`` column layout — so no axis move is needed. Both
``vert_difference_option`` branches are checked, with the tendencies seeded to
nonzero inputs so the accumulation (``dt_* = dt_* - ...``) is exercised.
"""

from functools import partial
from pathlib import Path

import jax
import numpy as np
import pytest

import jsca  # noqa: F401
from jsca.dycore import four_in_one

FIXTURE = Path(__file__).parent / "fixtures" / "four_in_one_reference.npz"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="four_in_one fixtures not generated"
)

OPTIONS = {"sb": "simmons_and_burridge", "mcm": "mcm"}


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _call(fx, option):
    pk, bk = fx["fio_pk"], fx["fio_bk"]
    dpk, dbk = np.diff(pk), np.diff(bk)
    return four_in_one(
        fx["fio_divg"], fx["fio_u_grid"], fx["fio_v_grid"], fx["fio_t_grid"],
        fx["fio_p_surf"], fx["fio_ln_p_half"], fx["fio_ln_p_full"], fx["fio_p_full"],
        fx["fio_dx_psg"], fx["fio_dy_psg"],
        fx["fio_dt_psg_in"], fx["fio_dt_tg_in"], fx["fio_dt_ug_in"], fx["fio_dt_vg_in"],
        dpk, dbk, bk, vert_difference_option=option,
    )


@pytest.mark.parametrize("tag", list(OPTIONS))
def test_four_in_one(fx, tag):
    dt_psg, dt_tg, dt_ug, dt_vg, wg, wg_full = _call(fx, OPTIONS[tag])
    # arithmetic with a log-derived input (ln_p_*) -> rtol 1e-13
    for got, ref in [
        (dt_psg, f"fio_{tag}_dt_psg"), (dt_tg, f"fio_{tag}_dt_tg"),
        (dt_ug, f"fio_{tag}_dt_ug"), (dt_vg, f"fio_{tag}_dt_vg"),
        (wg, f"fio_{tag}_wg"), (wg_full, f"fio_{tag}_wg_full"),
    ]:
        np.testing.assert_allclose(np.asarray(got), fx[ref], rtol=1e-13, atol=1e-15)


def test_wg_zero_at_boundaries(fx):
    """wg (interface mass flux) is zero at the model top and the surface (F90 L1108-1109)."""
    *_, wg, _ = _call(fx, "simmons_and_burridge")
    wg = np.asarray(wg)
    assert np.all(wg[..., 0] == 0.0) and np.all(wg[..., -1] == 0.0)


def test_jit(fx):
    """jit (vert_difference_option static) matches eager."""
    pk, bk = fx["fio_pk"], fx["fio_bk"]
    dpk, dbk = np.diff(pk), np.diff(bk)
    jitted = jax.jit(partial(four_in_one, vert_difference_option="simmons_and_burridge"))
    out = jitted(
        fx["fio_divg"], fx["fio_u_grid"], fx["fio_v_grid"], fx["fio_t_grid"],
        fx["fio_p_surf"], fx["fio_ln_p_half"], fx["fio_ln_p_full"], fx["fio_p_full"],
        fx["fio_dx_psg"], fx["fio_dy_psg"],
        fx["fio_dt_psg_in"], fx["fio_dt_tg_in"], fx["fio_dt_ug_in"], fx["fio_dt_vg_in"],
        dpk, dbk, bk,
    )
    np.testing.assert_allclose(np.asarray(out[1]), fx["fio_sb_dt_tg"], rtol=1e-13, atol=1e-15)
