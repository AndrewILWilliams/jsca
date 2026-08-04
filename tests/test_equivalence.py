import numpy as np

from jsca.testing import benjamini_hochberg, ensemble_mean_test, ks_distribution_test


def make_members(rng, n_members, shape, loc=0.0, scale=1.0):
    return rng.normal(loc=loc, scale=scale, size=(n_members, *shape))


def test_same_distribution_passes():
    rng = np.random.default_rng(0)
    ctrl = make_members(rng, 5, (20, 30))
    test = make_members(rng, 5, (20, 30))
    res = ensemble_mean_test(ctrl, test, alpha=0.05)
    assert res.fail_fraction < 0.01  # FDR keeps false alarms near zero


def test_shifted_region_detected():
    rng = np.random.default_rng(1)
    ctrl = make_members(rng, 5, (20, 30))
    test = make_members(rng, 5, (20, 30))
    test[:, :10, :] += 10.0  # 10-sigma shift in half the domain
    res = ensemble_mean_test(ctrl, test, alpha=0.05)
    assert res.reject[:10, :].mean() > 0.9
    assert res.reject[10:, :].mean() < 0.05
    assert np.all(res.delta[:10, :] > 5.0)


def test_floor_suppresses_tiny_differences():
    rng = np.random.default_rng(2)
    ctrl = make_members(rng, 5, (500,), scale=1e-4)  # tiny internal variability
    test = make_members(rng, 5, (500,), scale=1e-4) + 1e-3  # "significant" but tiny
    strict = ensemble_mean_test(ctrl, test, alpha=0.05, floor=0.0)
    floored = ensemble_mean_test(ctrl, test, alpha=0.05, floor=0.01)
    assert strict.fail_fraction > 0.5  # statistically detectable…
    assert floored.fail_fraction == 0.0  # …but below practical significance


def test_benjamini_hochberg_known_case():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.5, 0.9])
    reject = benjamini_hochberg(p, alpha=0.05)
    # largest i with p_(i) <= 0.05*i/10 is i=2 (0.008 <= 0.010)
    assert reject.sum() == 2
    assert reject[:2].all() and not reject[2:].any()


def test_ks_same_vs_shifted():
    rng = np.random.default_rng(3)
    a = rng.normal(size=2000)
    stat_same, p_same = ks_distribution_test(a, rng.normal(size=2000))
    stat_shift, p_shift = ks_distribution_test(a, rng.normal(loc=0.5, size=2000))
    assert p_same > 0.01
    assert p_shift < 1e-6
    assert stat_shift > stat_same
