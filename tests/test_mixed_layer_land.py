"""Tests for the ``land_option='input'`` surface fields of the mixed layer.

The land bucket test case gives land its own thermal inertia and albedo via two
masked scalings in ``mixed_layer_init`` (``mixed_layer.F90`` L433/L437 for albedo,
L514/L554 for heat capacity): ``field = base``, then ``where(land) field =
prefactor * base``. These are pure elementwise arithmetic (no lookup, no log/exp)
so they need no golden ``.npz`` — the identity is checked directly here, and the
full land-coupled slab is validated end-to-end against Isca at T21 in the bucket
climatology run.

We also pin that :func:`mixed_layer_step` with a uniform ``heat_capacity`` field
reproduces the scalar-``depth`` ocean path bit-for-bit, so the existing Frierson
slab fixture (``test_mixed_layer_fixtures.py``) is unaffected by the
generalisation.
"""
import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca import constants
from jsca.physics import (
    MixedLayerParams,
    land_albedo,
    land_heat_capacity,
    mixed_layer_step,
)
from jsca.physics.vert_diff import TriSurf


def test_land_heat_capacity_masked_scaling():
    land = np.array([True, False, True, False])
    depth, pref = 2.5, 0.1
    hc = np.asarray(land_heat_capacity(land, depth, pref))
    ocean = depth * constants.RHO_CP
    assert np.allclose(hc[~land], ocean, rtol=1e-14)
    assert np.allclose(hc[land], pref * ocean, rtol=1e-14)


def test_land_albedo_masked_scaling():
    land = np.array([True, False, True, False])
    albedo_value, pref = 0.25, 1.3
    alb = np.asarray(land_albedo(land, albedo_value, pref))
    assert np.allclose(alb[~land], albedo_value, rtol=1e-14)
    assert np.allclose(alb[land], pref * albedo_value, rtol=1e-14)


def test_prefactor_one_is_identity():
    """A prefactor of 1 leaves land indistinguishable from ocean."""
    land = np.array([True, False, True])
    hc = np.asarray(land_heat_capacity(land, 2.5, 1.0))
    assert np.allclose(hc, 2.5 * constants.RHO_CP, rtol=1e-14)
    alb = np.asarray(land_albedo(land, 0.31, 1.0))
    assert np.allclose(alb, 0.31, rtol=1e-14)


def test_uniform_heat_capacity_field_matches_scalar_path():
    """Passing depth*RHO_CP as a field reproduces the default scalar ocean step."""
    rng = np.random.default_rng(0)
    n = 6
    z = np.zeros(n)
    tri = TriSurf(e=z, f_t=z, f_q=z,
                  delta_t=rng.standard_normal(n) * 1e-3,
                  delta_q=rng.standard_normal(n) * 1e-6,
                  mu_delt_n=np.full(n, -3.0), nu_n=z, e_n1=z,
                  dflux=rng.standard_normal(n) * 1e-2)
    params = MixedLayerParams(depth=2.5, albedo=0.31, evaporation=True)
    args = (params, np.full(n, 300.0), rng.standard_normal(n) * 10,
            np.abs(rng.standard_normal(n)) * 1e-4, np.full(n, 400.0),
            np.full(n, 300.0), np.full(n, 200.0), np.full(n, 5.0),
            np.full(n, 1e-6), np.full(n, 2.0), np.full(n, 1.0),
            np.full(n, -1e-6), tri, 720.0)

    scalar = mixed_layer_step(*args)
    field = mixed_layer_step(*args, heat_capacity=np.full(n, 2.5 * constants.RHO_CP))
    assert np.allclose(np.asarray(scalar[0]), np.asarray(field[0]), rtol=1e-15, atol=0)
    assert np.allclose(np.asarray(scalar[1]), np.asarray(field[1]), rtol=1e-15, atol=0)


@pytest.mark.parametrize("pref", [0.1, 0.5, 2.0])
def test_land_capacity_changes_sst_increment(pref):
    """A smaller land heat capacity gives a larger SST increment for the same flux."""
    n = 3
    z = np.zeros(n)
    tri = TriSurf(e=z, f_t=z, f_q=z, delta_t=z, delta_q=z,
                  mu_delt_n=np.full(n, -3.0), nu_n=z, e_n1=z, dflux=z)
    params = MixedLayerParams(depth=2.5, albedo=0.31, evaporation=True)
    land = np.array([True, True, True])
    common = (params, np.full(n, 300.0), np.full(n, 10.0), np.full(n, 1e-4),
              np.full(n, 400.0), np.full(n, 300.0), np.full(n, 200.0),
              np.full(n, 5.0), np.full(n, 1e-6), np.full(n, 2.0),
              np.full(n, 1.0), np.full(n, -1e-6), tri, 720.0)
    ocean = mixed_layer_step(*common)
    land_step = mixed_layer_step(
        *common, heat_capacity=land_heat_capacity(land, params.depth, pref))
    ocean_dts = abs(float(ocean[1][0]))
    land_dts = abs(float(land_step[1][0]))
    if pref < 1.0:
        assert land_dts > ocean_dts
    else:
        assert land_dts < ocean_dts
