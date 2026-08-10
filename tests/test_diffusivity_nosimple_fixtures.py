"""Tier-1 tests for the boundary-layer diffusivity on Isca's DEFAULT path
(``diffusivity_nml do_simple=.false.``) against real-Fortran fixtures.

Isca's ``diffusivity_nml`` default is ``do_simple=.false.`` -- what the column and
Frierson runs actually use (neither sets ``diffusivity_nml``). It differs from the
``do_simple=.true.`` path in two ways:

* the dry-static-energy ``svcp`` carries the **virtual-temperature** correction
  ``T*(1+d608*q)`` (so ``q`` matters), and
* **unstable** columns (``b_star > 0``) place the PBL top with a **parcel-buoyancy**
  crossing (``svcp > svp``) instead of the bulk-Richardson one.

Fixtures come from ``fortran_instrumentation/dump_diffusivity_nosimple_reference.F90``
(the unmodified ``diffusivity.F90`` with ``do_simple=.false.``, real ``monin_obukhov``
kernel, a moist-near-surface humidity profile, and ``b_star`` swept from stable
(-0.010) to unstable (+0.045)) -- so both PBL branches and the virtual term are
exercised. Pure arithmetic, so the diffusivities and PBL depth match to machine
precision.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics import DiffusivityParams, diffusivity

FIXTURE = Path(__file__).parent / "fixtures" / "diffusivity_nosimple_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="do_simple=.false. diffusivity fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def result(fx):
    # DiffusivityParams() defaults to do_simple=False (Isca's default), matching
    # the fixture; q enters the virtual-temperature svcp.
    return [np.asarray(a) for a in diffusivity(
        DiffusivityParams(), fx["df_t"], fx["df_q"],
        fx["df_u"], fx["df_v"], fx["df_z_full"], fx["df_z_half"],
        fx["df_u_star"], fx["df_b_star"])]


def test_spans_both_pbl_branches(fx):
    """Guard: the fixture actually exercises stable and unstable columns."""
    assert fx["df_b_star"].min() < 0.0 < fx["df_b_star"].max()


def test_pbl_depth_matches_fortran(fx, result):
    # exact: the PBL walk is pure interpolation (both Richardson and parcel branches)
    _, _, h = result
    assert np.allclose(h, fx["df_h"], rtol=1e-13, atol=1e-12)


def test_k_m_matches_fortran(fx, result):
    k_m, _, _ = result
    assert np.allclose(k_m, fx["df_k_m"], rtol=1e-12, atol=1e-12)


def test_k_t_matches_fortran(fx, result):
    _, k_t, _ = result
    assert np.allclose(k_t, fx["df_k_t"], rtol=1e-12, atol=1e-12)


def test_differs_from_do_simple(fx, result):
    """The virtual term / unstable branch genuinely change the answer vs do_simple
    (else the default choice would be moot)."""
    _, _, h_ns = result
    _, _, h_simple = [np.asarray(a) for a in diffusivity(
        DiffusivityParams(do_simple=True), fx["df_t"], fx["df_q"],
        fx["df_u"], fx["df_v"], fx["df_z_full"], fx["df_z_half"],
        fx["df_u_star"], fx["df_b_star"])]
    assert not np.allclose(h_ns, h_simple, rtol=1e-3)
