"""Tier-1 tests for the Held-Suarez forcing against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_hs_forcing_reference.F90``,
which compiles Isca's actual ``hs_forcing.F90`` unmodified against no-op stubs
for its FMS infrastructure (the default ``Held_Suarez`` path exercises none of
that machinery). Regeneration recipe in that file's header.

Fortran storage is ``(lon, lat, level)`` with level LAST — exactly the port's
``(..., K)`` column layout — so no axis move is needed; ``lat`` is the 2-D
horizontal field. The fixture validates the composite ``(udt, vdt, tdt)``
returned by the public ``hs_forcing`` (Rayleigh drag; Newtonian + frictional
heating). ``teq`` is a Fortran-internal diagnostic, implied by ``tdt``.
"""

from functools import partial
from pathlib import Path

import jax
import numpy as np
import pytest

import jsca  # noqa: F401
from jsca.physics import hs_forcing, hs_forcing_init, newtonian_damping, rayleigh_damping

FIXTURE = Path(__file__).parent / "fixtures" / "hs_forcing_reference.npz"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="hs_forcing fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _run(fx):
    dt = float(fx["hs_meta"][3])
    p = hs_forcing_init()  # ka=-40, ks=-4, kf=-1 (Isca defaults)
    udt, vdt, tdt, teq = hs_forcing(
        p, fx["hs_lat"], fx["hs_p_half"], fx["hs_p_full"],
        fx["hs_u"], fx["hs_v"], fx["hs_t"], fx["hs_um"], fx["hs_vm"], dt,
    )
    return udt, vdt, tdt


def test_hs_forcing_composite(fx):
    udt, vdt, tdt = _run(fx)
    # log / exp (** kappa) bearing -> rtol 1e-13 (rule 2)
    np.testing.assert_allclose(np.asarray(udt), fx["hs_udt"], rtol=1e-13, atol=1e-20)
    np.testing.assert_allclose(np.asarray(vdt), fx["hs_vdt"], rtol=1e-13, atol=1e-20)
    np.testing.assert_allclose(np.asarray(tdt), fx["hs_tdt"], rtol=1e-13, atol=1e-18)


def test_pbl_where_clause_exercised(fx):
    """The fixture spans the boundary layer, so both branches of the damping
    where-clause (sigma > sigma_b vs not) actually fire."""
    sigma = fx["hs_p_full"] / fx["hs_p_half"][..., -1:]
    assert np.any(sigma > 0.7) and np.any(sigma <= 0.7)


def test_split_matches_composite(fx):
    """newtonian_damping + rayleigh_damping + frictional heating reproduce the
    composite tdt (cross-check the exposed sub-routines)."""
    dt = float(fx["hs_meta"][3])
    p = hs_forcing_init()
    ps = fx["hs_p_half"][..., -1]
    utnd, vtnd = rayleigh_damping(p, ps, fx["hs_p_full"], fx["hs_u"], fx["hs_v"])
    ttnd, _teq = newtonian_damping(p, fx["hs_lat"], ps, fx["hs_p_full"], fx["hs_t"])
    fric = -((fx["hs_um"] + 0.5 * np.asarray(utnd) * dt) * np.asarray(utnd)
             + (fx["hs_vm"] + 0.5 * np.asarray(vtnd) * dt) * np.asarray(vtnd)) / p.cp_air
    np.testing.assert_allclose(np.asarray(ttnd) + fric, fx["hs_tdt"], rtol=1e-13, atol=1e-18)


def test_jit(fx):
    """jit (params static) matches eager."""
    dt = float(fx["hs_meta"][3])
    p = hs_forcing_init()
    jitted = jax.jit(partial(hs_forcing, p))
    udt, vdt, tdt, _ = jitted(
        fx["hs_lat"], fx["hs_p_half"], fx["hs_p_full"],
        fx["hs_u"], fx["hs_v"], fx["hs_t"], fx["hs_um"], fx["hs_vm"], dt,
    )
    np.testing.assert_allclose(np.asarray(tdt), fx["hs_tdt"], rtol=1e-13, atol=1e-18)
