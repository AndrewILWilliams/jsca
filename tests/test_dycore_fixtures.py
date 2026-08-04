"""Tier-1 tests for the time-stepping spine against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_dycore_reference.F90``,
which compiles Isca's actual ``leapfrog.F90``, ``matrix_invert.F90`` and
``press_and_geopot.F90`` unmodified. Regeneration recipe in that file's
header. Index arguments in fixtures are 1-based (Fortran); tests convert.

Tolerances: leapfrog is pure arithmetic in matching op order → near-bitwise.
press_and_geopot involves log/exp → limited by libm differences (~1e-15
relative per call). matrix_invert deliberately uses LAPACK instead of the
Fortran's Gauss–Jordan (documented deviation) → 1e-11 on well-conditioned
semi-implicit-like matrices.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import jsca  # noqa: F401  (x64)
from jsca.dycore import (
    compute_geopotential,
    compute_pressures_and_heights,
    compute_z_bot,
    invert,
    leapfrog,
    leapfrog_2level_a,
    leapfrog_2level_a_real,
    leapfrog_2level_b,
    pressure_variables,
)

FIXTURE = Path(__file__).parent / "fixtures" / "dycore_reference.npz"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="dycore fixtures not generated")


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def cplx(fx, stem):
    return jnp.asarray(fx[f"{stem}_re"] + 1j * fx[f"{stem}_im"])


# ------------------------------ leapfrog -------------------------------------


def test_leapfrog_combined_normal_step(fx):
    a_in, dta = cplx(fx, "lf_a_in"), cplx(fx, "lf_dta")
    dt, robert, raw = fx["lf_scalars"]
    out = leapfrog(a_in, dta, 0, 1, 2, dt, robert, raw)  # Fortran (1,2,3)
    np.testing.assert_allclose(np.asarray(out), cplx(fx, "lf_combined_out"), rtol=1e-14)


def test_leapfrog_combined_first_step(fx):
    a_in, dta = cplx(fx, "lf_a_in"), cplx(fx, "lf_dta")
    dt, robert, raw = fx["lf_scalars"]
    out = leapfrog(a_in, dta, 0, 0, 1, dt, robert, raw)  # Fortran (1,1,2)
    np.testing.assert_allclose(np.asarray(out), cplx(fx, "lf_first_out"), rtol=1e-14)


def test_leapfrog_two_level_split(fx):
    a_in, dta = cplx(fx, "lf_a_in"), cplx(fx, "lf_dta")
    dt, robert, raw = fx["lf_scalars"]
    a1, part = leapfrog_2level_a(a_in, dta, 0, 1, 2, dt, robert, raw)
    np.testing.assert_allclose(np.asarray(a1), cplx(fx, "lf_2lvA_out"), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(part), cplx(fx, "lf_2lvA_part"), rtol=1e-14)
    a1 = a1.at[..., 2].multiply(1.01)  # the driver's stand-in implicit correction
    a2 = leapfrog_2level_b(a1, part, 1, 2, robert, raw)
    np.testing.assert_allclose(np.asarray(a2), cplx(fx, "lf_2lvB_out"), rtol=1e-14)


def test_leapfrog_two_level_a_real_variant_quirk(fx):
    """The real variant omits raw_filter_coeff on the current level (Fortran quirk)."""
    a_in, dta = jnp.asarray(fx["lf_r_a_in"]), jnp.asarray(fx["lf_r_dta"])
    dt, robert, raw = fx["lf_scalars"]
    a1, part = leapfrog_2level_a_real(a_in, dta, 0, 1, 2, dt, robert, raw)
    np.testing.assert_allclose(np.asarray(a1), fx["lf_r_2lvA_out"], rtol=1e-14)
    np.testing.assert_allclose(np.asarray(part), fx["lf_r_2lvA_part"], rtol=1e-14)
    # and it really is different from the complex-variant semantics:
    a1c, _ = leapfrog_2level_a(a_in.astype(complex), dta.astype(complex), 0, 1, 2, dt, robert, raw)
    assert not np.allclose(np.asarray(a1c).real, fx["lf_r_2lvA_out"], rtol=1e-10)


# ---------------------------- matrix_invert ----------------------------------


@pytest.mark.parametrize("n", [4, 25])
def test_invert_matches_fortran(fx, n):
    m_in = fx[f"mi_in_{n}"]
    inv, det = invert(m_in)
    np.testing.assert_allclose(np.asarray(inv), fx[f"mi_inv_{n}"], rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(float(det), float(fx[f"mi_det_{n}"][0]), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(inv) @ m_in, np.eye(n), atol=1e-12)


# --------------------------- press_and_geopot --------------------------------


def test_pressure_variables_uneven_sigma(fx):
    pk, bk, ps = fx["pg_pk"], fx["pg_bk"], fx["pg1_ps"]
    p_half, ln_p_half, p_full, ln_p_full = pressure_variables(pk, bk, ps)
    np.testing.assert_allclose(np.asarray(p_half), fx["pg1_p_half"], rtol=1e-14)
    np.testing.assert_allclose(np.asarray(ln_p_half), fx["pg1_ln_p_half"], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(np.asarray(p_full), fx["pg1_p_full"], rtol=1e-14)
    np.testing.assert_allclose(np.asarray(ln_p_full), fx["pg1_ln_p_full"], rtol=1e-14)


def test_compute_geopotential_dry(fx):
    pk = fx["pg_pk"]
    gfull, ghalf = compute_geopotential(
        pk, fx["pg1_t"], fx["pg1_ln_p_half"], fx["pg1_ln_p_full"], fx["pg1_phis"]
    )
    np.testing.assert_allclose(np.asarray(ghalf), fx["pg1_geopot_half"], rtol=1e-13)
    np.testing.assert_allclose(np.asarray(gfull), fx["pg1_geopot_full"], rtol=1e-13)


def test_compute_z_bot_dry(fx):
    z = compute_z_bot(fx["pg_pk"], fx["pg_bk"], fx["pg1_ps"], fx["pg1_t"][:, :, -1])
    np.testing.assert_allclose(np.asarray(z), fx["pg1_z_bot"], rtol=1e-13)


def test_compute_pressures_and_heights_dry(fx):
    zf, zh, pf, ph = compute_pressures_and_heights(
        fx["pg_pk"], fx["pg_bk"], fx["pg1_t"], fx["pg1_ps"], fx["pg1_phis"]
    )
    np.testing.assert_allclose(np.asarray(zf), fx["pg1_z_full"], rtol=1e-13)
    np.testing.assert_allclose(np.asarray(zh), fx["pg1_z_half"], rtol=1e-13)
    np.testing.assert_allclose(np.asarray(pf), fx["pg1_p_full"], rtol=1e-14)
    np.testing.assert_allclose(np.asarray(ph), fx["pg1_p_half"], rtol=1e-14)


def test_pressure_variables_hybrid_top(fx):
    """Nonzero model-top pressure exercises the general (non-sigma) branch."""
    pk2, bk2, ps = fx["pg2_pk"], fx["pg2_bk"], fx["pg1_ps"]
    p_half, ln_p_half, p_full, ln_p_full = pressure_variables(pk2, bk2, ps)
    np.testing.assert_allclose(np.asarray(p_half), fx["pg2_p_half"], rtol=1e-14)
    np.testing.assert_allclose(np.asarray(ln_p_half), fx["pg2_ln_p_half"], rtol=1e-14)
    np.testing.assert_allclose(np.asarray(p_full), fx["pg2_p_full"], rtol=1e-14)
    np.testing.assert_allclose(np.asarray(ln_p_full), fx["pg2_ln_p_full"], rtol=1e-14)


def test_compute_geopotential_virtual_temperature(fx):
    gfull, ghalf = compute_geopotential(
        fx["pg2_pk"],
        fx["pg1_t"],
        fx["pg2_ln_p_half"],
        fx["pg2_ln_p_full"],
        fx["pg1_phis"],
        q_grid=fx["pg2_q"],
        use_virtual_temperature=True,
    )
    np.testing.assert_allclose(np.asarray(ghalf), fx["pg2_geopot_half"], rtol=1e-13)
    np.testing.assert_allclose(np.asarray(gfull), fx["pg2_geopot_full"], rtol=1e-13)


def test_compute_z_bot_virtual(fx):
    z = compute_z_bot(
        fx["pg2_pk"],
        fx["pg2_bk"],
        fx["pg1_ps"],
        fx["pg1_t"][:, :, -1],
        qg=fx["pg2_q"][:, :, -1],
        use_virtual_temperature=True,
    )
    np.testing.assert_allclose(np.asarray(z), fx["pg2_z_bot"], rtol=1e-13)
