"""Validate GammaGuard's NaI(Tl) peak fitting against the claims of Breitenmoser et al. (2022).

Data: ETH Research Collection, doi:10.3929/ethz-b-000528920 (downloaded on demand, not redistributed).
Claims tested for each of the four crystals:
  1. H = d1 * E (linear, no offset) describes the calibration, R^2 > 0.9999, relative LOO error < 0.02 %
  2. FWHM(H) = b1 * H^b2 describes the resolution, R^2 > 0.997, relative LOO error < 0.2 %
Lines (gross spectra, >= 300 keV; lower-energy lines sit on X-ray / scatter structure): see LINES.

    python validation/validate_nai_rll.py [--json validation/results/nai_rll.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from gammaguard.datasets import ETH_RLL, load_rll
from gammaguard.datasets.eth_rll import RLL_NOMINAL
from gammaguard.qa.measure import measure_lines
from gammaguard.qa.profiles import NAI

LINES = {
    "Unat": [351.932, 609.320],
    "Cs137": [661.657],
    "Y88": [898.042, 1836.06],
    "Co60": [1173.228, 1332.492],
    "Knat": [1460.822],
    "Thnat": [2614.51],
}


def measure_crystal(crystal):
    """Fit every line in every source file for one crystal (two passes, widths seeded by the first)."""
    opts = NAI.fit
    rows = []
    for src, energies in LINES.items():
        spec = load_rll(src, "gr", crystal)
        kw = {"step": opts.step, "roi_fwhm": opts.roi_fwhm, "search_fwhm": opts.search_fwhm,
              "multiplet_fwhm": opts.multiplet_fwhm,
              "group_fwhm_keV": lambda e: float(NAI.initial_resolution.fwhm(e))}
        first = measure_lines(spec.counts, RLL_NOMINAL, energies, lambda e: float(NAI.initial_resolution.fwhm(e)), **kw)
        seeds = {e: (f.fwhm * float(RLL_NOMINAL.slope(f.centroid)) if f else float(NAI.initial_resolution.fwhm(e)))
                 for e, f in zip(energies, first, strict=True)}
        fits = measure_lines(spec.counts, RLL_NOMINAL, energies, seeds.__getitem__, **kw)
        for e, f in zip(energies, fits, strict=True):
            rows.append({"source": src, "energy": e, "fit": f})
    return rows


def r2(y, yhat):
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)


def loo_rel_error(fit_fn, x, y, sig):
    """PRESS / total sum of squares, the paper's 'relative generalization error'."""
    press = 0.0
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        yhat = fit_fn(x[m], y[m], sig[m])(x[i])
        press += (y[i] - yhat) ** 2
    return press / np.sum((y - np.mean(y)) ** 2)


def fit_origin(x, y, sig):
    w = 1 / sig**2
    d = np.sum(w * x * y) / np.sum(w * x * x)
    return lambda e: d * np.asarray(e)


def analyse(crystal):
    rows = [r for r in measure_crystal(crystal)]
    found = [r for r in rows if r["fit"] is not None]
    E = np.array([r["energy"] for r in found])
    H = np.array([r["fit"].centroid for r in found])
    Hs = np.array([r["fit"].centroid_err for r in found])
    W = np.array([r["fit"].fwhm for r in found])  # channels
    Ws = np.array([r["fit"].fwhm_err for r in found])

    # 1. calibration H = d1 E (paper) and H = h0 + d E (free offset)
    d1 = np.sum(E * H / Hs**2) / np.sum(E * E / Hs**2)
    A = np.stack([np.ones_like(E), E], axis=1) / Hs[:, None]
    coef, *_ = np.linalg.lstsq(A, H / Hs, rcond=None)
    cov = np.linalg.pinv(A.T @ A)
    # scale errors by the reduced chi^2 (model/peak-shape systematics dominate the tiny statistical errors)
    chi2 = np.sum(((H - (coef[0] + coef[1] * E)) / Hs) ** 2) / max(len(E) - 2, 1)
    off_ch, off_err_ch = coef[0], np.sqrt(cov[0, 0] * max(chi2, 1.0))
    offset_keV, offset_err_keV = -off_ch / coef[1], off_err_ch / coef[1]
    sys_sig = np.full_like(Hs, 0.05 * np.median(W))  # unweighted-ish LOO: 5% of FWHM per-line systematic
    cal_resid_keV = (H - d1 * E) / d1

    # 2. resolution FWHM(H) = b1 H^b2 (paper form), fitted in linear space
    model = lambda h, b1, b2: b1 * h**b2  # noqa: E731
    (b1, b2), _ = curve_fit(model, H, W, p0=(0.5, 0.5), sigma=np.maximum(Ws, 0.02 * W), maxfev=20000)

    def fit_res(x, y, s):
        p, _ = curve_fit(model, x, y, p0=(b1, b2), sigma=s, maxfev=20000)
        return lambda h: model(h, *p)

    sig_w = np.maximum(Ws, 0.02 * W)
    out = {
        "crystal": str(crystal),
        "n_lines": int(len(E)),
        "lines_not_found": [f"{r['source']} {r['energy']}" for r in rows if r["fit"] is None],
        "d1_ch_per_keV": float(d1),
        "kev_per_ch": float(1 / d1),
        "offset_free_fit_keV": float(offset_keV), "offset_err_keV": float(offset_err_keV),
        "free_fit_h0_ch": float(coef[0]), "free_fit_d_ch_per_keV": float(coef[1]),
        "cal_R2": float(r2(H, d1 * E)),
        "cal_max_abs_resid_keV": float(np.max(np.abs(cal_resid_keV))),
        "cal_rms_resid_keV": float(np.sqrt(np.mean(cal_resid_keV**2))),
        "cal_loo_rel_error": float(loo_rel_error(lambda x, y, s: fit_origin(x, y, s), E, H, sys_sig)),
        "b1": float(b1), "b2": float(b2),
        "res_R2": float(r2(W, model(H, b1, b2))),
        "res_loo_rel_error": float(loo_rel_error(fit_res, H, W, sig_w)),
        "fwhm_pct_at_662": float(100 * model(d1 * 661.657, b1, b2) / (d1 * 661.657)),
        "fwhm_pct_at_1332": float(100 * model(d1 * 1332.492, b1, b2) / (d1 * 1332.492)),
        "lines": [{"source": r["source"], "energy": r["energy"],
                   "read_keV": None if r["fit"] is None else float(r["fit"].centroid / d1),
                   "fwhm_pct": None if r["fit"] is None else float(100 * r["fit"].fwhm / r["fit"].centroid),
                   "reduced_deviance": None if r["fit"] is None else float(r["fit"].reduced_deviance)} for r in rows],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(Path(__file__).parent / "results" / "nai_rll.json"))
    args = ap.parse_args()
    results = [analyse(c) for c in (1, 2, 3, 4, "sum")]
    print(f"Dataset: {ETH_RLL.title}\n{ETH_RLL.citation}\n")
    hdr = f"{'crystal':<8}{'lines':>6}{'keV/ch':>8}{'offset (keV)':>16}{'cal R2':>10}{'cal LOO':>10}{'res R2':>9}{'res LOO':>9}{'b2':>7}{'FWHM662':>9}{'FWHM1332':>10}"
    print(hdr + "\n" + "-" * len(hdr))
    for r in results:
        print(f"{r['crystal']:<8}{r['n_lines']:>6}{r['kev_per_ch']:>8.4f}{r['offset_free_fit_keV']:>9.2f} +-{r['offset_err_keV']:<5.2f}"
              f"{r['cal_R2']:>10.6f}{100 * r['cal_loo_rel_error']:>9.4f}%{r['res_R2']:>9.4f}{100 * r['res_loo_rel_error']:>8.3f}%"
              f"{r['b2']:>7.3f}{r['fwhm_pct_at_662']:>8.2f}%{r['fwhm_pct_at_1332']:>9.2f}%")
    for r in results:
        if r["lines_not_found"]:
            print(f"crystal {r['crystal']}: lines not found: {r['lines_not_found']}")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"dataset": ETH_RLL.key, "results": results}, indent=1))
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
