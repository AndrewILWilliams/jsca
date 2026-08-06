"""Tier-1 tests for the top-of-model Rayleigh sponge against real Fortran.

Fixtures come from ``fortran_instrumentation/dump_damping_reference.F90``, which
compiles the **unmodified** ``damping_driver.f90`` and runs one Frierson sponge
step (``damping_driver_nml: do_rayleigh=.true., trayfric=-0.25,
sponge_pbottom=5000., do_conserve_energy=.true.``; all gravity-wave-drag paths
off) over a 4x6x25 grid. The reference pressures ``pref`` are the Frierson
pure-sigma levels; the 3-D ``p_full`` is that profile scaled per column so the
sponge's pressure threshold falls at different levels across the grid. Dumps the
u/v wind tendencies (relaxation toward zero) and the frictional heating.

The step is pure arithmetic (no lookup table, no log/exp), so there is no
documented deviation: everything matches to machine precision.

``nlev_rayfric`` is reconstructed in the test from the **same** dumped ``pref``
via :func:`damping_driver_init`, exactly as Isca's init does.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics import damping_driver_init, rayleigh_sponge

FIXTURE = Path(__file__).parent / "fixtures" / "damping_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="damping fixtures not generated"
)

DT = 720.0


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


@pytest.fixture(scope="module")
def result(fx):
    # nlev_rayfric + rfactr from the same reference profile the Fortran used.
    params = damping_driver_init(fx["dd_pref"], trayfric=-0.25,
                                 sponge_pbottom=5000.0, do_conserve_energy=True)
    udt, vdt, tdt = rayleigh_sponge(params, DT, fx["dd_pfull"], fx["dd_u"], fx["dd_v"])
    return {"udt": np.asarray(udt), "vdt": np.asarray(vdt), "tdt": np.asarray(tdt),
            "params": params}


@pytest.mark.parametrize("attr,key", [
    ("udt", "dd_udt"), ("vdt", "dd_vdt"), ("tdt", "dd_tdt"),
])
def test_matches_fortran(result, fx, attr, key):
    assert np.allclose(result[attr], fx[key], rtol=1e-12, atol=1e-18)


def test_nlev_rayfric(result, fx):
    """nlev_rayfric = level closest to 2*sponge_pbottom (=100 hPa here) => level 6."""
    assert result["params"].nlev_rayfric == 6


def test_physical_structure(result, fx):
    """Sponge decelerates the wind (opposes it) and dissipation heats (tdt >= 0)."""
    # tendency opposes the wind everywhere it acts
    acts = result["udt"] != 0.0
    assert np.all(result["udt"][acts] * fx["dd_u"][acts] <= 0.0)
    # frictional heating is non-negative, and confined to the sponge levels
    assert np.all(result["tdt"] >= -1e-18)
    # no damping below the sponge base pressure
    below = fx["dd_pfull"] >= 5000.0
    assert np.allclose(result["udt"][below], 0.0, atol=1e-18)
