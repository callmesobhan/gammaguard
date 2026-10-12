"""Monte Carlo benchmark: how often does each fault trip the check that should catch it, and
how often does a healthy detector raise a false alarm?

Every trial draws a *fresh* baseline (so baseline noise counts toward the false-alarm rate)
and a fresh test spectrum with the fault injected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qa import Baseline, Status, run_qa
from .simulate import (
    BackgroundScale,
    DeadChannel,
    DeadTime,
    Fault,
    GainDrift,
    HotChannel,
    NoiseBurst,
    RadonSurge,
    ResolutionLoss,
    SpuriousPeak,
    ZeroShift,
    background_scenario,
    inject,
    simulate,
    source_scenario,
)


@dataclass(frozen=True)
class Case:
    label: str
    mode: str  # "source" | "background"
    fault: Fault | None
    expected_check: str | None  # check that should fire; None for healthy detectors


CASES: tuple[Case, ...] = (
    Case("healthy", "source", None, None),
    Case("gain drift +0.01 %", "source", GainDrift(1e-4), "calibration"),
    Case("gain drift +0.03 %", "source", GainDrift(3e-4), "calibration"),
    Case("gain drift +0.10 %", "source", GainDrift(1e-3), "calibration"),
    Case("zero shift +0.05 keV", "source", ZeroShift(0.05), "calibration"),
    Case("zero shift +0.30 keV", "source", ZeroShift(0.30), "calibration"),
    Case("zero shift +1.00 keV", "source", ZeroShift(1.0), "calibration"),
    Case("resolution x1.04", "source", ResolutionLoss(1.04), "resolution"),
    Case("resolution x1.10", "source", ResolutionLoss(1.10), "resolution"),
    Case("resolution x1.25", "source", ResolutionLoss(1.25), "resolution"),
    Case("spurious 800 keV, 0.05 cps (30 ct)", "source", SpuriousPeak(800.0, 0.05), "unexpected_peaks"),
    Case("spurious 800 keV, 0.20 cps (120 ct)", "source", SpuriousPeak(800.0, 0.20), "unexpected_peaks"),
    Case("spurious 800 keV, 1.00 cps (600 ct)", "source", SpuriousPeak(800.0, 1.0), "unexpected_peaks"),
    Case("hot channel, +30 counts", "source", HotChannel(2200, 0.05), "channel_anomalies"),
    Case("hot channel, +300 counts", "source", HotChannel(2200, 0.5), "channel_anomalies"),
    Case("dead channel at 400 keV (~37 ct/ch)", "source", DeadChannel(1000), "channel_anomalies"),
    Case("dead channel at 880 keV (~10 ct/ch)", "source", DeadChannel(2200), "channel_anomalies"),
    Case("dead time 35 %", "source", DeadTime(0.35), "acquisition"),
    Case("healthy (background run)", "background", None, None),
    Case("background x1.10", "background", BackgroundScale(1.10), "background"),
    Case("background x1.50", "background", BackgroundScale(1.50), "background"),
    Case("background x2.50", "background", BackgroundScale(2.50), "background"),
    Case("radon progeny x3", "background", RadonSurge(3.0), "background"),
    Case("electronic noise 0.5 cps", "background", NoiseBurst(0.5), "background"),
    Case("spurious 700 keV, 0.01 cps (36 ct)", "background", SpuriousPeak(700.0, 0.01), "unexpected_peaks"),
    Case("spurious 700 keV, 0.03 cps (108 ct)", "background", SpuriousPeak(700.0, 0.03), "unexpected_peaks"),
)


NAI_CASES: tuple[Case, ...] = (
    Case("healthy", "source", None, None),
    Case("gain drift +0.2 %", "source", GainDrift(2e-3), "calibration"),
    Case("gain drift +0.8 %", "source", GainDrift(8e-3), "calibration"),
    Case("gain drift +2.0 %", "source", GainDrift(2e-2), "calibration"),
    Case("zero shift +1 keV", "source", ZeroShift(1.0), "calibration"),
    Case("zero shift +8 keV", "source", ZeroShift(8.0), "calibration"),
    Case("zero shift +20 keV", "source", ZeroShift(20.0), "calibration"),
    Case("resolution x1.04", "source", ResolutionLoss(1.04), "resolution"),
    Case("resolution x1.12", "source", ResolutionLoss(1.12), "resolution"),
    Case("resolution x1.30", "source", ResolutionLoss(1.30), "resolution"),
    Case("spurious 2200 keV, 0.3 cps (180 ct)", "source", SpuriousPeak(2200.0, 0.3), "unexpected_peaks"),
    Case("spurious 2200 keV, 1.5 cps (900 ct)", "source", SpuriousPeak(2200.0, 1.5), "unexpected_peaks"),
    Case("hot channel, +180 counts", "source", HotChannel(650, 0.3), "channel_anomalies"),
    Case("hot channel, +1200 counts", "source", HotChannel(650, 2.0), "channel_anomalies"),
    Case("dead channel at 180 keV (~100 ct/ch)", "source", DeadChannel(60), "channel_anomalies"),
    Case("dead time 35 %", "source", DeadTime(0.35), "acquisition"),
    Case("healthy (background run)", "background", None, None),
    Case("background x1.10", "background", BackgroundScale(1.10), "background"),
    Case("background x1.50", "background", BackgroundScale(1.50), "background"),
    Case("background x2.50", "background", BackgroundScale(2.50), "background"),
    Case("radon progeny x3", "background", RadonSurge(3.0), "background"),
    Case("electronic noise 3 cps", "background", NoiseBurst(3.0), "background"),
)

CASES_BY_DETECTOR: dict[str, tuple[Case, ...]] = {"hpge": CASES, "nai": NAI_CASES}


@dataclass(frozen=True)
class CaseResult:
    case: Case
    trials: int
    p_warn_or_worse: float  # overall report
    p_fail: float
    p_expected_check: float | None  # expected check at WARN or worse (None for healthy cases)


def run_case(case: Case, trials: int, seed: int, detector: str = "hpge", source_time: float = 600.0,
             bkg_time: float = 3600.0) -> CaseResult:
    ss = np.random.SeedSequence(seed)
    n_warn = n_fail = n_expected = 0
    for child in ss.spawn(trials):
        r_base_s, r_base_b, r_test = (np.random.default_rng(s) for s in child.spawn(3))
        baseline = Baseline.build(
            simulate(source_scenario(1800.0, detector), r_base_s),
            simulate(background_scenario(4 * 3600.0, detector), r_base_b),
            profile=detector,
        )
        if case.mode == "source":
            sc = source_scenario(source_time, detector)
        else:
            sc = background_scenario(bkg_time, detector)
        if case.fault is not None:
            sc = inject(sc, case.fault)
        report = run_qa(baseline, simulate(sc, r_test), mode=case.mode)
        n_warn += report.status >= Status.WARN
        n_fail += report.status >= Status.FAIL
        if case.expected_check:
            n_expected += report.check(case.expected_check).status >= Status.WARN
    return CaseResult(case, trials, n_warn / trials, n_fail / trials,
                      n_expected / trials if case.expected_check else None)


def run_benchmark(trials: int = 20, seed: int = 2026, cases: tuple[Case, ...] | None = None,
                  detector: str = "hpge") -> list[CaseResult]:
    cases = CASES_BY_DETECTOR[detector] if cases is None else cases
    return [run_case(c, trials, seed + i, detector) for i, c in enumerate(cases)]


def format_results(results: list[CaseResult]) -> str:
    head = f"{'case':<40}{'mode':<12}{'expected check':<20}{'caught':>8}{'any WARN+':>11}{'any FAIL':>10}"
    lines = [head, "-" * len(head)]
    for r in results:
        caught = "-" if r.p_expected_check is None else f"{r.p_expected_check:.0%}"
        lines.append(f"{r.case.label:<40}{r.case.mode:<12}{r.case.expected_check or '-':<20}"
                     f"{caught:>8}{r.p_warn_or_worse:>11.0%}{r.p_fail:>10.0%}")
    n = results[0].trials if results else 0
    lines.append(f"\n{n} trials per case; each trial uses a fresh baseline and a fresh spectrum.")
    return "\n".join(lines)
