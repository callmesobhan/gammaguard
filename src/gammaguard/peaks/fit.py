"""Peak fitting by Poisson maximum likelihood: one or several Gaussians on a smooth background.

The fit is done in *channel* space on purpose: the whole point of the QA checks is that the
energy calibration may be wrong, so peak positions must not depend on it.

Model on a region of interest (ROI)::

    m(x) = sum_i A_i * [Phi((x+1/2-mu_i)/sigma_i) - Phi((x-1/2-mu_i)/sigma_i)]   Gaussians (exact per-channel integral)
         + b0 + b1 * (x - x_c)                                                   linear background
         + sum_i S_i * Phi((mu_i - x)/sigma_i)                                   optional Compton step (low side)

with ``sigma_i = s * sigma0_i``: the widths of the peaks in one fit share a single scale ``s``
relative to the expected widths ``sigma0_i``. For a lone peak that is simply a free width; for a
blended multiplet (Co-60 in NaI) it is what makes the fit well-posed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

from ..resolution import FWHM_PER_SIGMA
from ..stats.poisson import poisson_deviance_residuals


@dataclass(frozen=True)
class PeakFit:
    centroid: float  # channels
    centroid_err: float
    sigma: float  # channels
    sigma_err: float
    area: float  # counts
    area_err: float
    bkg_level: float  # counts / channel at the ROI centre
    bkg_slope: float  # counts / channel / channel
    deviance: float
    dof: int
    roi: tuple[int, int]  # [lo, hi) channels used
    step: float = 0.0  # fitted step height under this peak (counts / channel)
    group_size: int = 1  # number of peaks fitted together

    @property
    def fwhm(self) -> float:
        return self.sigma * FWHM_PER_SIGMA

    @property
    def fwhm_err(self) -> float:
        return self.sigma_err * FWHM_PER_SIGMA

    @property
    def reduced_deviance(self) -> float:
        return self.deviance / max(self.dof, 1)

    @property
    def significance(self) -> float:
        """Net area over its uncertainty."""
        return self.area / self.area_err if self.area_err > 0 else 0.0


def _model(theta, x, centre, n_peaks, sigma0, step):
    areas, mus = theta[:n_peaks], theta[n_peaks : 2 * n_peaks]
    s, b0, b1 = theta[2 * n_peaks : 2 * n_peaks + 3]
    m = b0 + b1 * (x - centre)
    for i in range(n_peaks):
        sig = s * sigma0[i]
        m = m + areas[i] * (ndtr((x + 0.5 - mus[i]) / sig) - ndtr((x - 0.5 - mus[i]) / sig))
        if step:
            m = m + theta[2 * n_peaks + 3 + i] * ndtr((mus[i] - x) / sig)
    return m


def fit_peaks(
    counts,
    centres: Sequence[float],
    fwhm_guess: Sequence[float],
    half_width_fwhm: float = 4.0,
    step: bool = False,
    scale_errors: bool = True,
) -> list[PeakFit] | None:
    """Fit peaks near ``centres`` (channels) simultaneously. ``None`` if the fit is not usable.

    ``fwhm_guess`` are the expected widths in channels; they set the starting point, the
    relative widths within the group, and the ROI. ``scale_errors`` inflates uncertainties by
    ``sqrt(deviance/dof)`` when the model fits worse than Poisson noise allows (Birge ratio), which
    keeps the QA thresholds honest on real data where the peak shape is only approximately Gaussian.
    """
    counts = np.asarray(counts, dtype=float)
    centres = np.asarray(centres, dtype=float)
    fw = np.asarray(fwhm_guess, dtype=float)
    n = len(centres)
    order = np.argsort(centres)
    centres, fw = centres[order], fw[order]

    hw = max(int(np.ceil(half_width_fwhm * fw.max())), 8)
    lo = max(int(round(centres[0])) - hw, 0)
    hi = min(int(round(centres[-1])) + hw + 1, counts.size)
    if hi - lo < 12:
        return None
    x = np.arange(lo, hi, dtype=float)
    y = counts[lo:hi]
    centre = 0.5 * (lo + hi - 1)

    edge = max(3, hw // 4)
    left, right = y[:edge].mean(), y[-edge:].mean()
    b1_0 = (right - left) / (x[-1] - x[0])
    b0_0 = max(0.5 * (left + right), 0.05)  # a zero start makes the Poisson likelihood singular
    sigma0 = fw / FWHM_PER_SIGMA
    net = (y - (b0_0 + b1_0 * (x - centre))).clip(min=0)

    areas0, mus0 = [], []
    for c, f in zip(centres, fw, strict=True):
        near = np.abs(x - c) <= f
        a0 = max(float(net[near].sum()), 1.0)
        areas0.append(a0)
        if n == 1 and a0 > 1.0:  # a lone peak: start at its centroid
            w = net[np.abs(x - c) <= 1.5 * f]
            mus0.append(float(np.sum(x[np.abs(x - c) <= 1.5 * f] * w) / max(w.sum(), 1e-9)))
        else:
            mus0.append(float(c))

    span = max(float(y.max()), 1.0)
    theta0 = list(areas0) + mus0 + [1.0, b0_0, b1_0]
    lower = [0.0] * n + [m - 2.0 * f for m, f in zip(mus0, fw, strict=True)] + [0.35, 1e-3, -span]
    upper = [10.0 * a + 10.0 * span for a in areas0] + [m + 2.0 * f for m, f in zip(mus0, fw, strict=True)]
    upper += [3.0, span, span]
    if step:
        theta0 += [0.0] * n
        lower += [0.0] * n
        upper += [span] * n
    theta0 = np.clip(theta0, np.array(lower) + 1e-9, np.array(upper) - 1e-9)

    def resid(theta):
        return poisson_deviance_residuals(y, _model(theta, x, centre, n, sigma0, step))

    try:
        sol = least_squares(resid, theta0, bounds=(lower, upper), x_scale="jac", method="trf", max_nfev=1000)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not sol.success:
        return None

    deviance = float(np.sum(sol.fun**2))
    dof = max(len(y) - len(theta0), 1)
    cov = np.linalg.pinv(sol.jac.T @ sol.jac)
    err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    if scale_errors:
        err = err * max(1.0, np.sqrt(deviance / dof))
    s, b0, b1 = sol.x[2 * n : 2 * n + 3]
    s_err = err[2 * n]
    fits = []
    for i in range(n):
        area, mu = sol.x[i], sol.x[n + i]
        a_err, mu_err = err[i], err[n + i]
        if not (np.isfinite([a_err, mu_err, s_err]).all() and area > 0 and a_err > 0 and mu_err > 0 and s_err > 0):
            return None
        fits.append(PeakFit(
            centroid=float(mu), centroid_err=float(mu_err),
            sigma=float(s * sigma0[i]), sigma_err=float(s_err * sigma0[i]),
            area=float(area), area_err=float(a_err),
            bkg_level=float(b0), bkg_slope=float(b1),
            deviance=deviance, dof=dof, roi=(lo, hi),
            step=float(sol.x[2 * n + 3 + i]) if step else 0.0, group_size=n,
        ))
    # return in the order the caller gave
    out: list[PeakFit | None] = [None] * n
    for rank, idx in enumerate(order):
        out[idx] = fits[rank]
    return out  # type: ignore[return-value]


def fit_peak(
    counts,
    centre_guess: float,
    fwhm_guess: float,
    half_width_fwhm: float = 4.0,
    scale_errors: bool = True,
    step: bool = False,
) -> PeakFit | None:
    """Fit a single peak near ``centre_guess`` (channels). ``None`` if the fit is not usable."""
    fits = fit_peaks(counts, [centre_guess], [fwhm_guess], half_width_fwhm, step, scale_errors)
    return None if fits is None else fits[0]
