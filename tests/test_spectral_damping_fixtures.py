"""Tier-1 tests for spectral_damping against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_spectral_damping_reference.F90``,
which compiles Isca's actual ``spectral_damping.F90`` unmodified (transforms_mod
stubbed to supply ``get_eigen_laplacian``/``get_spec_domain``; the stub
replicates ``spherical.F90``'s own eigenvalue formula). Regeneration recipe in
that file's header.

Three scenarios exercise the three ``damping_option`` branches (rd / ri / exp),
each running the generic 2-D and 3-D compute plus the vorticity and divergence
routines (which add the top-level eddy / zonal-mean sponges). Fortran spectral
storage ``(m, n, k)`` maps directly onto the port's layout; complex arrays are
dumped as separate ``_re``/``_im`` halves.

Tolerances (CLAUDE.md rule 2): the resolution-dependent/independent tables carry
an integer power of ``eigen`` and the exponential branch carries ``exp/log`` →
log/exp-bearing, rtol 1e-13. The eigen cross-check against
``jsca.grid.spectral`` and the pure-arithmetic sponge structure are 1e-14.
"""

from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.dycore import (
    compute_spectral_damping,
    compute_spectral_damping_div,
    compute_spectral_damping_vor,
    spectral_damping_init,
)
from jsca.grid.spectral import Truncation, laplacian_eigenvalues

FIXTURE = Path(__file__).parent / "fixtures" / "spectral_damping_reference.npz"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="spectral_damping fixtures not generated"
)

RADIUS = 6376.0e3  # matches the driver's radius (jsca EARTH_RADIUS)

# damping_option parameters, mirroring dump_spectral_damping_reference.F90.
SCENARIOS = {
    "rd": dict(
        damping_coeff=1.157e-04,
        damping_order=4,
        damping_option="resolution_dependent",
        cutoff_wn=15,
        eddy_sponge_coeff=5.0e-05,
        zmu_sponge_coeff=1.5e-04,
        zmv_sponge_coeff=2.5e-04,
        damping_coeff_vor=2.3e-04,
        damping_order_vor=4,
        damping_coeff_div=3.1e-04,
        damping_order_div=2,
        damping_coeff_r=1.0e-06,
    ),
    "ri": dict(
        damping_coeff=1.0e18,
        damping_order=2,
        damping_option="resolution_independent",
        cutoff_wn=15,
        eddy_sponge_coeff=5.0e-05,
        zmu_sponge_coeff=1.5e-04,
        zmv_sponge_coeff=2.5e-04,
        damping_coeff_vor=2.0e18,
        damping_order_vor=2,
        damping_coeff_div=1.5e12,
        damping_order_div=1,
    ),
    "exp": dict(
        damping_coeff=1.0e-04,
        damping_order=2,
        damping_option="exponential_cutoff",
        cutoff_wn=15,
        eddy_sponge_coeff=5.0e-05,
        zmu_sponge_coeff=1.5e-04,
        zmv_sponge_coeff=2.5e-04,
        damping_coeff_vor=2.0e-04,
        damping_order_vor=3,
        damping_coeff_div=3.0e-04,
        damping_order_div=1,
    ),
}


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def cplx(fx, stem):
    return fx[f"{stem}_re"] + 1j * fx[f"{stem}_im"]


def build(fx, tag):
    return spectral_damping_init(eigen=fx["sd_eigen"], **SCENARIOS[tag])


# --------------------------- eigen provenance --------------------------------


def test_eigen_matches_jsca_grid(fx):
    """The Fortran eigen (positive l(l+1)/a^2) is jsca's negated Laplacian eigenvalue."""
    nf, ns = int(fx["sd_meta"][0]), int(fx["sd_meta"][1])
    assert fx["sd_eigen"].shape == (nf + 1, ns + 1)
    expected = -laplacian_eigenvalues(Truncation(nf), RADIUS)
    np.testing.assert_allclose(fx["sd_eigen"], expected, rtol=1e-14, atol=0.0)


# ---------------------------- generic compute --------------------------------


@pytest.mark.parametrize("tag", ["rd", "ri", "exp"])
def test_compute_generic_3d(fx, tag):
    p = build(fx, tag)
    dt = float(fx["sd_meta"][4])
    out = compute_spectral_damping(p, cplx(fx, "sd_f3"), cplx(fx, "sd_g3"), dt)
    np.testing.assert_allclose(np.asarray(out), cplx(fx, f"sd_{tag}_gen3"), rtol=1e-13)


@pytest.mark.parametrize("tag", ["rd", "ri", "exp"])
def test_compute_generic_2d(fx, tag):
    p = build(fx, tag)
    dt = float(fx["sd_meta"][4])
    spec2, dtspec2 = cplx(fx, "sd_f3")[:, :, 0], cplx(fx, "sd_g3")[:, :, 0]
    out = compute_spectral_damping(p, spec2, dtspec2, dt)
    np.testing.assert_allclose(np.asarray(out), cplx(fx, f"sd_{tag}_gen2"), rtol=1e-13)


# ----------------------- vorticity / divergence + sponge ---------------------


@pytest.mark.parametrize("tag", ["rd", "ri", "exp"])
def test_compute_vor(fx, tag):
    p = build(fx, tag)
    dt = float(fx["sd_meta"][4])
    out = compute_spectral_damping_vor(p, cplx(fx, "sd_f3"), cplx(fx, "sd_g3"), dt)
    np.testing.assert_allclose(np.asarray(out), cplx(fx, f"sd_{tag}_vor"), rtol=1e-13)


@pytest.mark.parametrize("tag", ["rd", "ri", "exp"])
def test_compute_div(fx, tag):
    p = build(fx, tag)
    dt = float(fx["sd_meta"][4])
    out = compute_spectral_damping_div(p, cplx(fx, "sd_f3"), cplx(fx, "sd_g3"), dt)
    np.testing.assert_allclose(np.asarray(out), cplx(fx, f"sd_{tag}_div"), rtol=1e-13)


def test_vor_div_differ_from_generic_on_top_level(fx):
    """Sanity: the sponge actually changes the first level of vor/div vs generic."""
    p = build(fx, "rd")
    dt = float(fx["sd_meta"][4])
    gen = np.asarray(compute_spectral_damping(p, cplx(fx, "sd_f3"), cplx(fx, "sd_g3"), dt))
    vor = np.asarray(
        compute_spectral_damping_vor(p, cplx(fx, "sd_f3"), cplx(fx, "sd_g3"), dt)
    )
    # generic uses `damping`, vor uses `damping_vor` + sponge → level 0 must differ.
    assert not np.allclose(gen[:, :, 0], vor[:, :, 0])


# --------------------------- init-table structure ----------------------------


def test_exponential_flag_and_zero_below_cutoff(fx):
    """Exponential option zeroes the generic damping at/below the cutoff (F90 L138-147)."""
    p = build(fx, "exp")
    assert p.exponential is True
    eigen = fx["sd_eigen"]
    cutoff = SCENARIOS["exp"]["cutoff_wn"]
    below = eigen / eigen[0, cutoff] <= 1.0
    np.testing.assert_array_equal(np.asarray(p.damping)[below], 0.0)
    assert np.all(np.asarray(p.damping)[~below] > 0.0)


def test_resolution_dependent_linear_drag(fx):
    """damping_coeff_r adds a uniform floor to the generic table (F90 L157-159)."""
    p = build(fx, "rd")
    assert float(np.min(np.asarray(p.damping))) >= SCENARIOS["rd"]["damping_coeff_r"]
