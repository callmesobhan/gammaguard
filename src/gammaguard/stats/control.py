"""Statistical process control for QA time series (one value per routine check).

A single-run tolerance check only fires once a drift has already crossed the limit.
CUSUM / EWMA on the run-to-run series fire on small *sustained* shifts well before that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControlResult:
    statistic: np.ndarray
    alarm_index: int | None  # first index at which the chart signals, if any


def standardise(values, n_reference: int) -> np.ndarray:
    """z-scores using mean/std of the first ``n_reference`` (assumed in-control) points."""
    v = np.asarray(values, dtype=float)
    ref = v[:n_reference]
    sd = ref.std(ddof=1)
    if sd <= 0:
        raise ValueError("reference period has zero variance")
    return (v - ref.mean()) / sd


def shewhart(z, limit: float = 3.0) -> ControlResult:
    """Individuals chart: signal when ``|z| > limit``."""
    z = np.asarray(z, dtype=float)
    hits = np.flatnonzero(np.abs(z) > limit)
    return ControlResult(z, int(hits[0]) if hits.size else None)


def cusum(z, k: float = 0.5, h: float = 5.0) -> ControlResult:
    """Two-sided tabular CUSUM on z-scores. The statistic is ``max(C+, C-)``."""
    z = np.asarray(z, dtype=float)
    c_pos = c_neg = 0.0
    out = np.empty_like(z)
    alarm = None
    for i, x in enumerate(z):
        c_pos = max(0.0, c_pos + x - k)
        c_neg = max(0.0, c_neg - x - k)
        out[i] = max(c_pos, c_neg)
        if alarm is None and out[i] > h:
            alarm = i
    return ControlResult(out, alarm)


def ewma(z, lam: float = 0.2, width: float = 3.0) -> ControlResult:
    """EWMA chart with exact time-varying control limits for unit-variance inputs."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    prev = 0.0
    alarm = None
    for i, x in enumerate(z):
        prev = lam * x + (1.0 - lam) * prev
        out[i] = prev
        limit = width * np.sqrt(lam / (2.0 - lam) * (1.0 - (1.0 - lam) ** (2 * (i + 1))))
        if alarm is None and abs(prev) > limit:
            alarm = i
    return ControlResult(out, alarm)
