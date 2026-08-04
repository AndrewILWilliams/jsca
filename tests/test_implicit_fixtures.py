"""Tier-1 tests for the semi-implicit correction against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_implicit_reference.F90``, which
compiles Isca's actual ``implicit.F90`` unmodified, linked against the real
``press_and_geopot.F90`` and ``matrix_invert.F90`` (transforms_mod stubbed for
``get_spec_domain``). Regeneration recipe in that file's header.

Only ``implicit_init``/``implicit_correction``/``implicit_end`` are public in the
Fortran, so the reference-state matrices, the linear operators and
``adjust_dt_divs`` are all validated *transitively* through
``implicit_correction`` outputs. Four scenarios cover both vertical-difference
options and the sigma (``pk = 0``) / hybrid (``pk /= 0``) branches, at two
timesteps.

Tolerance: the correction routes the divergence tendency through the inverse
vertical matrices, which the port builds with LAPACK rather than the Fortran's
Gauss–Jordan (documented deviation in ``matrix_invert.py``), so outputs are
inversion-limited — rtol 1e-10. Fortran ``previous``/``current`` are 1-based;
tests convert to 0-based.
"""

from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.dycore import build_wave_matrices, implicit_correction, implicit_init

FIXTURE = Path(__file__).parent / "fixtures" / "implicit_reference.npz"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="implicit fixtures not generated")

TAGS = ["a1", "a2", "b", "c"]
# Inversion-limited: the port builds the vertical matrices with LAPACK, not the
# Fortran's Gauss–Jordan (documented deviation, matrix_invert.py). Achieved error
# on these small well-conditioned matrices is ~1e-15; held at the documented 1e-11.
RTOL = 1e-11
ATOL = 1e-11


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def cplx(fx, stem):
    return fx[f"{stem}_re"] + 1j * fx[f"{stem}_im"]


def _run(fx, tag):
    meta = fx[f"imp_{tag}_meta"]
    num_total = int(meta[4])
    alpha, dt, ref_surf_p = float(meta[5]), float(meta[6]), float(meta[7])
    previous, current = int(meta[8]) - 1, int(meta[9]) - 1  # Fortran 1-based -> 0-based
    vopt = "mcm" if tag == "b" else "simmons_and_burridge"

    p = implicit_init(
        pk=fx[f"imp_{tag}_pk"],
        bk=fx[f"imp_{tag}_bk"],
        ref_temperature=fx[f"imp_{tag}_ref_t"],
        ref_surf_p=ref_surf_p,
        num_total_wavenumbers=num_total,
        eigen=fx[f"imp_{tag}_eigen"],
        wavenumber=fx[f"imp_{tag}_wavenum"].astype(np.int64),
        alpha=alpha,
        vert_difference_option=vopt,
    )
    wave = build_wave_matrices(p, dt)
    return p, implicit_correction(
        p, wave, dt,
        cplx(fx, f"imp_{tag}_dt_divs_in"),
        cplx(fx, f"imp_{tag}_dt_ts_in"),
        cplx(fx, f"imp_{tag}_dt_ln_ps_in"),
        cplx(fx, f"imp_{tag}_divs"),
        cplx(fx, f"imp_{tag}_ts"),
        cplx(fx, f"imp_{tag}_ln_ps"),
        previous, current,
    )


@pytest.mark.parametrize("tag", TAGS)
def test_dt_divs(fx, tag):
    _, (dt_divs, _, _) = _run(fx, tag)
    np.testing.assert_allclose(
        np.asarray(dt_divs), cplx(fx, f"imp_{tag}_dt_divs_out"), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("tag", TAGS)
def test_dt_ts(fx, tag):
    _, (_, dt_ts, _) = _run(fx, tag)
    np.testing.assert_allclose(
        np.asarray(dt_ts), cplx(fx, f"imp_{tag}_dt_ts_out"), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("tag", TAGS)
def test_dt_ln_ps(fx, tag):
    _, (_, _, dt_ln_ps) = _run(fx, tag)
    np.testing.assert_allclose(
        np.asarray(dt_ln_ps), cplx(fx, f"imp_{tag}_dt_ln_ps_out"), rtol=RTOL, atol=ATOL
    )


def test_eigen_provenance(fx):
    """The driver's eigen table is the positive l(l+1)/a^2 = -jsca Laplacian eigenvalue."""
    from jsca.grid.spectral import Truncation, laplacian_eigenvalues

    meta = fx["imp_a1_meta"]
    nf = int(meta[0])
    expected = -laplacian_eigenvalues(Truncation(nf), 6376.0e3)
    np.testing.assert_allclose(fx["imp_a1_eigen"], expected, rtol=1e-14, atol=0.0)


def test_wave_matrix_l0_is_identity(fx):
    """factor=0 at l=0, so the l=0 vertical matrix inverts to the identity."""
    p, _ = _run(fx, "a1")
    wave = build_wave_matrices(p, float(fx["imp_a1_meta"][6]))
    k = p.ref_t.shape[0]
    np.testing.assert_allclose(np.asarray(wave[0]), np.eye(k), atol=1e-12)
