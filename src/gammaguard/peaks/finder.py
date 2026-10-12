"""Peak search with a Poisson-aware second-difference filter (Mariscotti-style).

For each channel the counts are correlated with a zero-sum, Gaussian-shaped second-derivative
kernel matched to the local peak width. That rejects constant and linear backgrounds. The
variance of the filter output is propagated from the counts (``sum k_j^2 n_j``), so the
significance is meaningful in sparse regions where ``sqrt(n)`` shortcuts fail.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from ..resolution import FWHM_PER_SIGMA

WidthSpec = float | Callable[[float], float]


@dataclass(frozen=True)
class PeakCandidate:
    channel: int
    significance: float


def _kernel(fwhm_ch: float) -> np.ndarray:
    sigma = max(fwhm_ch, 1.5) / FWHM_PER_SIGMA
    half = max(3, int(np.ceil(4.0 * sigma)))
    x = np.arange(-half, half + 1, dtype=float)
    k = (1.0 - x**2 / sigma**2) * np.exp(-(x**2) / (2.0 * sigma**2))
    return k - k.mean()  # exact zero sum -> constant backgrounds cancel


def significance_curve(counts, fwhm_ch: WidthSpec, band: int = 256) -> np.ndarray:
    """Filter output divided by its Poisson standard deviation, per channel."""
    x = np.asarray(counts, dtype=float)
    n = x.size
    sig = np.zeros(n)
    for lo in range(0, n, band):
        hi = min(lo + band, n)
        width = fwhm_ch(0.5 * (lo + hi)) if callable(fwhm_ch) else fwhm_ch
        k = _kernel(float(width))
        pad = len(k) // 2
        a, b = max(0, lo - pad), min(n, hi + pad)
        seg = x[a:b]
        s = np.convolve(seg, k, mode="same")
        var = np.convolve(seg, k**2, mode="same")
        var = np.maximum(var, 0.5 * np.sum(k**2))  # floor: a single count is not a peak
        full = s / np.sqrt(var)
        sig[lo:hi] = full[lo - a : lo - a + (hi - lo)]
        # zero-padding at the true array ends creates artificial edges
        if lo == 0:
            sig[: min(pad, hi)] = 0.0
        if hi == n:
            sig[max(hi - pad, lo) : hi] = 0.0
    return sig


def find_peak_candidates(
    counts, fwhm_ch: WidthSpec, threshold: float = 5.0, min_channel: int = 0, band: int = 256
) -> list[PeakCandidate]:
    """Local maxima of the significance curve above ``threshold`` (channels >= ``min_channel``)."""
    sig = significance_curve(counts, fwhm_ch, band)
    sig[:min_channel] = 0.0
    width = fwhm_ch(0.5 * len(sig)) if callable(fwhm_ch) else fwhm_ch
    idx, _ = find_peaks(sig, height=threshold, distance=max(int(0.7 * width), 2))
    return [PeakCandidate(int(i), float(sig[i])) for i in idx]
