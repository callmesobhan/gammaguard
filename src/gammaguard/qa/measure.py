"""Locate and fit the reference lines in a spectrum, given an approximate calibration."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.signal import medfilt

from ..calibration import EnergyCalibration
from ..peaks.finder import significance_curve
from ..peaks.fit import PeakFit, fit_peaks


def group_lines(
    energies: Sequence[float], fwhm_keV: Callable[[float], float], multiplet_fwhm: float
) -> list[list[int]]:
    """Indices (into ``energies``) of lines that must be fitted together: neighbours closer than
    ``multiplet_fwhm`` times the FWHM at the lower line are blended."""
    order = sorted(range(len(energies)), key=lambda i: energies[i])
    groups: list[list[int]] = []
    for i in order:
        if groups and energies[i] - energies[groups[-1][-1]] < multiplet_fwhm * fwhm_keV(energies[groups[-1][-1]]):
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def measure_group(
    counts,
    calibration: EnergyCalibration,
    energies: Sequence[float],
    fwhm_keV: Sequence[float],
    search_fwhm: float = 10.0,
    min_significance: float = 5.0,
    step: bool = False,
    roi_fwhm: float = 4.0,
) -> list[PeakFit | None]:
    """Fit one group of lines (one line, or a blended multiplet) located near where the
    calibration says they are. The group is first positioned by scanning one common channel
    shift within ``+-search_fwhm`` FWHM (matched-filter score); individual centroids are then free
    in the fit."""
    counts = np.asarray(counts)
    ch0 = np.array([float(calibration.channel(e)) for e in energies])
    fwhm_ch = np.array([f / float(calibration.slope(c)) for f, c in zip(fwhm_keV, ch0, strict=True)])
    half = int(np.ceil(search_fwhm * fwhm_ch.max()))
    lo, hi = max(int(ch0.min()) - half, 0), min(int(ch0.max()) + half + 1, counts.size)
    if hi - lo < 12:
        return [None] * len(energies)
    # Score candidate positions with the zero-sum matched filter, not raw counts: on a scintillator the
    # smoothed continuum rises towards low energy and would win over the peak.
    # A 3-point median first removes single-channel spikes (hot channels), which would otherwise win.
    smooth = significance_curve(medfilt(counts.astype(float), 3), float(fwhm_ch.mean()), band=max(32, counts.size // 8))
    shifts = np.arange(-half, half + 1)
    score = np.zeros(shifts.size)
    for c in ch0:
        idx = np.clip(np.round(c + shifts).astype(int), 0, counts.size - 1)
        score += smooth[idx]
    shift = int(shifts[int(np.argmax(score))])
    fits = fit_peaks(counts, ch0 + shift, fwhm_ch, half_width_fwhm=roi_fwhm, step=step)
    if fits is None:
        return [None] * len(energies)
    out: list[PeakFit | None] = []
    for fit, c, w in zip(fits, ch0 + shift, fwhm_ch, strict=True):
        ok = fit is not None and fit.significance >= min_significance and abs(fit.centroid - c) <= 2.0 * w
        out.append(fit if ok else None)
    return out


def measure_lines(
    counts,
    calibration: EnergyCalibration,
    energies: Sequence[float],
    fwhm_keV: Callable[[float], float],
    search_fwhm: float = 10.0,
    step: bool = False,
    multiplet_fwhm: float = 2.5,
    roi_fwhm: float = 4.0,
    group_fwhm_keV: Callable[[float], float] | None = None,
) -> list[PeakFit | None]:
    """Fit every reference line; blended lines are fitted together. Result is aligned with
    ``energies``; ``None`` marks a line with no significant peak where expected.

    ``group_fwhm_keV`` decides which lines are blended. It should be a smooth resolution law, not
    the per-line seeds, otherwise a pair near the threshold can be grouped differently from one
    pass to the next (default: ``fwhm_keV``)."""
    out: list[PeakFit | None] = [None] * len(energies)
    for group in group_lines(energies, group_fwhm_keV or fwhm_keV, multiplet_fwhm):
        es = [energies[i] for i in group]
        fits = measure_group(
            counts, calibration, es, [fwhm_keV(e) for e in es], search_fwhm, step=step, roi_fwhm=roi_fwhm
        )
        for i, fit in zip(group, fits, strict=True):
            out[i] = fit
    return out


def measure_line(
    counts,
    calibration: EnergyCalibration,
    energy: float,
    fwhm_keV: float,
    search_fwhm: float = 10.0,
    min_significance: float = 5.0,
    step: bool = False,
    roi_fwhm: float = 4.0,
) -> PeakFit | None:
    """Single-line convenience wrapper around `measure_group`."""
    return measure_group(counts, calibration, [energy], [fwhm_keV], search_fwhm, min_significance, step, roi_fwhm)[0]
