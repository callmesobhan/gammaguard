import numpy as np
import pytest

from gammaguard.simulate import (
    DeadChannel,
    GainDrift,
    HotChannel,
    ResolutionLoss,
    ZeroShift,
    expected_counts,
    inject,
    parse_fault,
    simulate,
    source_scenario,
)


def _centroid(counts, ch0, half=15):
    x = np.arange(ch0 - half, ch0 + half + 1)
    y = counts[ch0 - half : ch0 + half + 1]
    y = y - np.median(counts[ch0 - 40 : ch0 - 20])
    return float(np.sum(x * y) / np.sum(y))


def test_simulation_is_reproducible_and_poisson():
    sc = source_scenario()
    a, b = simulate(sc, 7), simulate(sc, 7)
    assert (a.counts == b.counts).all()
    mu = expected_counts(sc)
    assert a.total_counts == pytest.approx(mu.sum(), rel=0.01)


def test_nothing_below_lld():
    sc = source_scenario()
    mu = expected_counts(sc)
    assert mu[: int(sc.detector.lld_keV / 0.4) - 1].sum() == 0


def test_gain_drift_moves_peaks_by_fraction():
    base = source_scenario()
    drifted = inject(base, GainDrift(0.001))
    ch0 = int(round(float(base.detector.calibration.channel(661.657))))
    shift = _centroid(expected_counts(drifted), ch0 + 3) - _centroid(expected_counts(base), ch0 + 3)
    assert shift == pytest.approx(ch0 * 0.001, abs=0.05)
    assert drifted.faults == ("gain_drift(fraction=0.001)",)


def test_zero_shift_reads_high_by_given_kev():
    sc = inject(source_scenario(), ZeroShift(0.5))
    true_cal, nominal = sc.detector.true_calibration, sc.detector.calibration
    ch = 1654.0
    assert nominal(ch) - true_cal(ch) == pytest.approx(0.5)


def test_resolution_loss_widens_peaks():
    sc = inject(source_scenario(), ResolutionLoss(1.5))
    assert sc.detector.resolution.fwhm(1332.5) == pytest.approx(
        1.5 * source_scenario().detector.resolution.fwhm(1332.5))


def test_channel_faults():
    sc = inject(source_scenario(), HotChannel(2200, 0.5), DeadChannel(1000))
    mu = expected_counts(sc)
    assert mu[2200] > 250 and mu[1000] == 0.0


def test_parse_fault_cli_syntax():
    assert parse_fault("gain_drift:0.0005") == GainDrift(0.0005)
    assert parse_fault("hot_channel:2200:0.5").channel == 2200
    with pytest.raises(ValueError):
        parse_fault("nonsense:1")
    with pytest.raises(ValueError):
        parse_fault("hot_channel:2200")
