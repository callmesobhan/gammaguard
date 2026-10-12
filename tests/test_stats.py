import numpy as np
import pytest

from gammaguard.stats import (
    compare_rates,
    currie_limits,
    cusum,
    ewma,
    poisson_deviance_residuals,
    shewhart,
    standardise,
)


def test_deviance_residuals_zero_when_perfect_and_signed():
    assert poisson_deviance_residuals([5, 10], [5, 10]) == pytest.approx([0.0, 0.0])
    r = poisson_deviance_residuals([0, 20], [4, 10])
    assert r[0] < 0 < r[1]
    assert r[0] ** 2 == pytest.approx(2 * 4)  # n = 0: deviance is 2 m


def test_deviance_is_chi_square_like_for_poisson_data():
    rng = np.random.default_rng(0)
    mu = np.full(20000, 50.0)
    dev = np.sum(poisson_deviance_residuals(rng.poisson(mu), mu) ** 2)
    assert dev / mu.size == pytest.approx(1.0, abs=0.05)


def test_compare_rates_equal_rates_not_significant():
    c = compare_rates(100, 100.0, 400, 400.0)
    assert c.ratio == pytest.approx(1.0)
    assert c.p_value > 0.9
    assert c.ci_low < 1.0 < c.ci_high


def test_compare_rates_detects_doubling_with_unequal_live_times():
    c = compare_rates(800, 100.0, 400, 100.0 * 4)  # 8 cps vs 1 cps
    assert c.ratio == pytest.approx(8.0)
    assert c.p_value < 1e-12


def test_compare_rates_handles_zero_counts():
    assert np.isnan(compare_rates(0, 1.0, 0, 1.0).ratio)
    c = compare_rates(0, 100.0, 50, 100.0)
    assert c.ratio == 0.0 and c.p_value < 1e-10


def test_currie_limits_known_values():
    lc, ld = currie_limits(100.0, paired=True)
    assert lc == pytest.approx(1.645 * np.sqrt(200))
    assert ld == pytest.approx(2.706 + 2 * lc, rel=1e-3)
    lc2, _ = currie_limits(100.0, paired=False)
    assert lc2 == pytest.approx(16.45)


def test_cusum_detects_small_sustained_shift_sooner_than_shewhart():
    """Median detection delay over many runs: a 1-sigma sustained shift starting at index 60."""
    delays = {"cusum": [], "ewma": [], "shewhart": []}
    for seed in range(200):
        x = np.random.default_rng(seed).normal(0, 1, 160)
        x[60:] += 1.0
        z = standardise(x, 40)
        for name, chart in (("cusum", cusum(z)), ("ewma", ewma(z)), ("shewhart", shewhart(z))):
            a = chart.alarm_index
            delays[name].append(np.inf if a is None or a < 60 else a - 60)  # ignore false alarms
    med = {k: float(np.median(v)) for k, v in delays.items()}
    assert med["cusum"] < 25
    assert med["ewma"] < 25
    assert med["cusum"] < med["shewhart"] and med["ewma"] < med["shewhart"]


def test_control_charts_quiet_on_in_control_data():
    z = standardise(np.random.default_rng(4).normal(0, 1, 60), 60)
    assert cusum(z, h=8.0).alarm_index is None
