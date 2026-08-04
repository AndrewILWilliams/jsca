"""Tier-3 statistical equivalence tools (scoping doc §4.5).

The contract: for each diagnostic, the JAX-minus-Fortran difference of
ensemble-mean climatologies must lie within the *Fortran* ensemble's
internal-variability envelope, with FDR control across the battery, plus
absolute practical floors so a wide envelope cannot hide real drift.

These functions are deliberately dumb about geoscience: they take arrays of
per-member climatologies and return per-point verdicts. Field handling,
area weighting, and the diagnostic battery live on top (Phase 2).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import stats


class EquivalenceResult(NamedTuple):
    delta: np.ndarray  # test-mean minus control-mean, per point
    se: np.ndarray  # sigma_ctrl * sqrt(2/N): SE of a difference of two N-means
    pvalue: np.ndarray  # two-sided t (df = N-1), per point
    reject: np.ndarray  # bool: significant after BH-FDR AND above floor
    fail_fraction: float  # fraction of points rejected


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BH false-discovery-rate control. Returns boolean 'reject' mask."""
    p = np.asarray(pvalues).ravel()
    m = p.size
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = p[order] <= thresh
    reject = np.zeros(m, dtype=bool)
    if below.any():
        kmax = np.max(np.nonzero(below)[0])
        reject[order[: kmax + 1]] = True
    return reject.reshape(np.shape(pvalues))


def ensemble_mean_test(
    control: np.ndarray,
    test: np.ndarray,
    alpha: float = 0.05,
    floor: float | np.ndarray = 0.0,
) -> EquivalenceResult:
    """Compare ensemble-mean climatologies against internal variability.

    Parameters
    ----------
    control, test : (N_members, ...) per-member climatological statistics
        (e.g. 5-year means). Member axis first.
    alpha : FDR level across all points.
    floor : practical-significance floor in the field's units; differences
        smaller than this are never rejected, however tight the envelope
        (guards against the opposite failure too: with tiny internal
        variability, meaningless differences would otherwise "fail").

    Notes
    -----
    Envelope uses the *control* ensemble spread only (the Fortran model
    defines truth and its own noise): SE = sigma_ctrl * sqrt(2/N), t with
    df = N-1. Conservative for small N; that is intentional.
    """
    control = np.asarray(control, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if control.shape[0] < 2:
        raise ValueError("need at least 2 ensemble members")
    n = control.shape[0]
    if test.shape[0] != n:
        raise ValueError("control and test must have the same ensemble size")

    delta = test.mean(axis=0) - control.mean(axis=0)
    sigma = control.std(axis=0, ddof=1)
    se = sigma * np.sqrt(2.0 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, delta / np.where(se > 0, se, 1.0), np.inf * np.sign(delta))
    pvalue = 2.0 * stats.t.sf(np.abs(t), df=n - 1)
    pvalue = np.where(np.isfinite(t), pvalue, np.where(np.abs(delta) > 0, 0.0, 1.0))

    reject = benjamini_hochberg(pvalue, alpha) & (np.abs(delta) > np.asarray(floor))
    return EquivalenceResult(
        delta=delta,
        se=se,
        pvalue=pvalue,
        reject=reject,
        fail_fraction=float(reject.mean()),
    )


def ks_distribution_test(control_samples: np.ndarray, test_samples: np.ndarray):
    """Two-sample KS on pooled scalar samples (e.g. daily global means).

    Returns (statistic, pvalue). Caveat for the battery: consecutive daily
    means are autocorrelated, which inflates KS significance — thin the
    series (e.g. every 10 days) or treat the p-value as an index, not a
    literal probability. The Phase-2 battery thins.
    """
    res = stats.ks_2samp(np.ravel(control_samples), np.ravel(test_samples))
    return float(res.statistic), float(res.pvalue)
