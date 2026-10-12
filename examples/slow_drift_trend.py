"""Slow gain drift: single-run tolerance checks vs. trend monitoring.

A detector's gain creeps upward from day 40 (e.g. a failing preamp or temperature control).
Each daily check is graded on its own (PASS until the drift crosses the tolerance) and also
fed into Shewhart / CUSUM / EWMA charts built from the first 30 in-control days.

    python examples/slow_drift_trend.py [--plot out.png]
"""

from __future__ import annotations

import argparse

import numpy as np

from gammaguard import Baseline, Status, run_qa
from gammaguard.simulate import GainDrift, background_scenario, inject, simulate, source_scenario
from gammaguard.stats import cusum, ewma, shewhart, standardise

DAYS, ONSET, RAMP_PER_DAY, REFERENCE_DAYS = 120, 40, 4e-6, 30


def main(plot: str | None = None, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    baseline = Baseline.build(simulate(source_scenario(1800.0), rng), simulate(background_scenario(4 * 3600.0), rng))

    shift, first_warn, first_fail = [], None, None
    for day in range(DAYS):
        drift = rng.normal(0, 1e-5) + RAMP_PER_DAY * max(0, day - ONSET)  # temperature jitter + ramp
        report = run_qa(baseline, simulate(inject(source_scenario(), GainDrift(drift)), rng))
        cal = report.check("calibration")
        shift.append(cal.details[-1]["shift_keV"])  # shift of the 1332.5 keV line vs baseline
        if first_warn is None and report.status >= Status.WARN:
            first_warn = day
        if first_fail is None and report.status >= Status.FAIL:
            first_fail = day

    z = standardise(shift, REFERENCE_DAYS)
    charts = {"Shewhart (3 sigma)": shewhart(z), "CUSUM (k=0.5, h=5)": cusum(z), "EWMA (lambda=0.2)": ewma(z)}
    print(f"drift starts on day {ONSET}; the 1332.5 keV line moves {RAMP_PER_DAY * 1332.5:.4f} keV/day vs baseline\n")
    print(f"{'method':<24}{'first alarm (day)':>18}{'shift then (keV)':>26}")
    rows = [("tolerance, single run: WARN", first_warn), ("tolerance, single run: FAIL", first_fail)]
    rows += [(name, c.alarm_index) for name, c in charts.items()]
    for name, day in rows:
        print(f"{name:<28}{'never' if day is None else day:>14}{'' if day is None else f'{shift[day]:>+22.3f}'}")

    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(shift, ".", ms=4, color="0.35", label="daily line shift vs baseline @ 1332.5 keV")
        tol = baseline.tolerances.shift_warn_fwhm * baseline.lines[-1].fwhm_keV
        ax.axhline(tol, color="tab:orange", ls="--", label="WARN tolerance")
        for name, c in charts.items():
            if c.alarm_index is not None:
                ax.axvline(c.alarm_index, ls=":", label=f"{name.split(' ')[0]} alarm (day {c.alarm_index})")
        ax.set(xlabel="day", ylabel="keV")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(plot, dpi=150)
        print(f"\nwrote {plot}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", help="save a PNG of the daily series and alarms")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    main(a.plot, a.seed)
