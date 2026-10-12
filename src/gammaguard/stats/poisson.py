"""Counting-statistics helpers. Exact Poisson/binomial tests are preferred over Gaussian
approximations because most of a gamma spectrum is in the low-count regime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def poisson_deviance_residuals(observed, expected):
    """Signed square roots of the per-bin Poisson deviance (``sum r^2`` is the deviance)."""
    n = np.asarray(observed, dtype=float)
    m = np.maximum(np.asarray(expected, dtype=float), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(n > 0, n * np.log(n / m), 0.0)
    d = np.maximum(2.0 * (term - (n - m)), 0.0)
    return np.sign(n - m) * np.sqrt(d)


@dataclass(frozen=True)
class RateComparison:
    """Two Poisson counts ``n1`` in ``t1`` s and ``n2`` in ``t2`` s: are the rates equal?"""

    n1: int
    t1: float
    n2: int
    t2: float
    ratio: float  # rate1 / rate2
    ci_low: float
    ci_high: float
    p_value: float


def compare_rates(n1: int, t1: float, n2: int, t2: float, confidence: float = 0.99) -> RateComparison:
    """Exact conditional test: given ``N = n1 + n2``, ``n1 ~ Binomial(N, t1 / (t1 + t2))`` under
    equal rates. Handles unequal live times and zero counts without any Gaussian assumption."""
    n_tot = int(n1 + n2)
    if n_tot == 0:
        return RateComparison(n1, t1, n2, t2, float("nan"), 0.0, float("inf"), 1.0)
    p0 = t1 / (t1 + t2)
    res = stats.binomtest(int(n1), n_tot, p0)
    lo, hi = res.proportion_ci(confidence_level=confidence)

    def to_ratio(p: float) -> float:
        if p >= 1.0:
            return float("inf")
        return (p / (1.0 - p)) * (t2 / t1)

    ratio = to_ratio(n1 / n_tot)
    return RateComparison(n1, t1, n2, t2, ratio, to_ratio(lo), to_ratio(hi), float(res.pvalue))


def currie_limits(background_counts: float, k: float = 1.645, paired: bool = True) -> tuple[float, float]:
    """Currie critical level ``L_C`` and detection limit ``L_D`` (net counts).

    ``paired=True``: background estimated from an equally long blank (``L_C = k sqrt(2B)``);
    ``paired=False``: background known exactly (``L_C = k sqrt(B)``).
    """
    b = float(background_counts)
    if paired:
        lc = k * np.sqrt(2.0 * b)
        return lc, k * k + 2.0 * lc
    lc = k * np.sqrt(b)
    return lc, k * k + 2.0 * lc


def bonferroni_alpha(alpha_family: float, n_tests: int) -> float:
    return alpha_family / max(int(n_tests), 1)
