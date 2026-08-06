"""Tier-1 tests for the slab-ocean surface energy balance against real Fortran.

Fixtures come from ``fortran_instrumentation/dump_mixed_layer_reference.F90``,
which compiles the **unmodified** ``mixed_layer.F90`` (+ ``vert_diff.F90`` for the
``surf_diff_type`` container) and runs one Frierson slab-ocean step
(``mixed_layer_nml: depth=2.5, albedo_value=0.31, evaporation=.true.``,
``land_option='none'`` — pure ocean, no q-flux, no prescribed SST) over a 4x6
grid with synthetic surface fluxes and vert_diff coupling terms. It dumps the
sea-surface-temperature update and the corrected lowest-level T/q increments the
up sweep needs.

The step is a closed arithmetic form (no lookup table, no log/exp), so there is
no documented deviation: everything matches to machine precision.

The ``mixed_layer`` step consumes and returns a
:class:`jsca.physics.vert_diff.TriSurf`; the fields it actually reads are
``mu_delt_n`` (Isca's ``dtmass``), ``dflux`` (the shared ``-nu*(1-e)``), and the
stored ``delta_t``/``delta_q``. The others are inert here and set to zero.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics import MixedLayerParams, mixed_layer_step
from jsca.physics.vert_diff import TriSurf

FIXTURE = Path(__file__).parent / "fixtures" / "mixed_layer_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="mixed_layer fixtures not generated"
)

DT = 720.0


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def result(fx):
    z = np.zeros_like(fx["ml_t_surf_in"])
    tri = TriSurf(
        e=z, f_t=z, f_q=z,
        delta_t=fx["ml_delta_t_in"], delta_q=fx["ml_delta_q_in"],
        mu_delt_n=fx["ml_dtmass"], nu_n=z, e_n1=z, dflux=fx["ml_dflux_t"],
    )
    params = MixedLayerParams(depth=2.5, albedo=0.31, evaporation=True)
    t_surf_new, delta_t_surf, tri_out = mixed_layer_step(
        params, fx["ml_t_surf_in"], fx["ml_flux_t"], fx["ml_flux_q"],
        fx["ml_flux_r"], fx["ml_net_sw"], fx["ml_lw_down"], fx["ml_dhdt_surf"],
        fx["ml_dedt_surf"], fx["ml_drdt_surf"], fx["ml_dhdt_atm"],
        fx["ml_dedq_atm"], tri, DT)
    return {
        "t_surf": np.asarray(t_surf_new),
        "delta_t": np.asarray(tri_out.delta_t),
        "delta_q": np.asarray(tri_out.delta_q),
    }


@pytest.mark.parametrize("attr,key", [
    ("t_surf", "ml_t_surf_out"),
    ("delta_t", "ml_delta_t_out"),
    ("delta_q", "ml_delta_q_out"),
])
def test_matches_fortran(result, fx, attr, key):
    assert np.allclose(result[attr], fx[key], rtol=1e-12, atol=1e-18)


def test_physical_structure(result, fx):
    """The SST update is a small, bounded increment (order 1e-2 K per 720 s)."""
    dts = result["t_surf"] - fx["ml_t_surf_in"]
    assert np.all(np.abs(dts) < 0.1)
