"""End-to-end QA on real NaI(Tl) spectra (ETH/PSI airborne-spectrometer data, Breitenmoser et al. 2022).

Experiment A  (detector-to-detector differences as real, independently measurable 'faults')
    A mixed check source is emulated by adding the Cs-137 and Co-60 rate spectra. A baseline is built
    from crystal 3; crystals 1, 2 and 4 (and 3 itself) are then graded against it. The gain and width
    differences reported by GammaGuard (3 lines) are compared with those predicted from the
    independent per-crystal calibration / resolution fits of validate_nai_rll.py (9 lines).
Experiment B  (background stability)
    The ten background measurements are graded in background mode against the first one.

    python validation/validate_nai_qa.py [--json validation/results/nai_qa.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from validate_nai_rll import analyse  # noqa: E402

from gammaguard import Baseline, run_qa  # noqa: E402
from gammaguard.datasets import fetch  # noqa: E402
from gammaguard.datasets.eth_rll import RLL_NOMINAL, SOURCES, effective_time, read_rate_csv  # noqa: E402
from gammaguard.spectrum import Spectrum  # noqa: E402

REFERENCE = (661.657, 1173.228, 1332.492)


def rates(source, kind, crystal):
    col = 1 if crystal == "sum" else 1 + int(crystal)
    _, arr = read_rate_csv(fetch("eth_rll", quiet=True) / f"RLL_data_{source}_{kind}.csv")
    return arr[:, col]


def to_spectrum(rate, **meta):
    t = effective_time(rate)
    return Spectrum(np.rint(rate * t).astype(np.int64), t, t, RLL_NOMINAL, meta)


def mixed_check_source(crystal):
    return to_spectrum(rates("Cs137", "gr", crystal) + rates("Co60", "gr", crystal), crystal=crystal)


def res_model(r):
    """FWHM in keV as a function of energy from the paper-form fit FWHM(H) = b1 H^b2, H = d1 E."""
    return lambda e: r["b1"] * (r["d1_ch_per_keV"] * e) ** r["b2"] / r["d1_ch_per_keV"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(Path(__file__).parent / "results" / "nai_qa.json"))
    args = ap.parse_args()
    ind = {str(r["crystal"]): r for r in (analyse(c) for c in (1, 2, 3, 4))}
    ref = "3"
    base = Baseline.build(mixed_check_source(ref), profile="nai", reference_energies=REFERENCE)
    print(f"Baseline from crystal {ref}: " + ", ".join(f"{ln.label} FWHM {100 * ln.fwhm_keV / ln.energy:.2f}%" for ln in base.lines))

    out = {"experiment_A": [], "experiment_B": []}
    print("\nExperiment A: crystals graded against crystal 3")
    print("  shift of each line relative to crystal 3 (keV):")
    print(f"{'crystal':<8}{'status':<7}{'line':>8}{'QA (3 lines)':>14}{'independent (9-line fits)':>27}{'FWHM ratio QA / indep':>24}")
    for c in ("3", "1", "2", "4"):
        rep = run_qa(base, mixed_check_source(c))
        cal, res = rep.check("calibration"), rep.check("resolution")
        ratio_ind = float(np.mean([res_model(ind[c])(e) / res_model(ind[ref])(e) for e in REFERENCE]))
        rows = []
        for d in cal.details:
            e = d["energy_keV"]
            # shift of the line relative to crystal 3, from the two crystals' smooth 9-line free-offset calibrations
            h_c = ind[c]["free_fit_h0_ch"] + ind[c]["free_fit_d_ch_per_keV"] * e
            h_r = ind[ref]["free_fit_h0_ch"] + ind[ref]["free_fit_d_ch_per_keV"] * e
            indep = (h_c - h_r) / ind[ref]["free_fit_d_ch_per_keV"]
            rows.append((e, d["shift_keV"], indep))
        out["experiment_A"].append({"crystal": c, "status": rep.status.label, "per_line_shift_keV": [
            {"energy": e, "qa": q, "independent": i} for e, q, i in rows], "fwhm_ratio_qa": res.metrics["fwhm_ratio_mean"],
            "fwhm_ratio_qa_err": res.metrics["fwhm_ratio_err"], "fwhm_ratio_independent": ratio_ind,
            "checks": {k.name: k.status.label for k in rep.checks}})
        for j, (e, q, i) in enumerate(rows):
            tail = f"{res.metrics['fwhm_ratio_mean']:>12.3f} / {ratio_ind:.3f}" if j == 0 else ""
            print(f"{c if j == 0 else '':<8}{rep.status.label if j == 0 else '':<7}{e:>8.1f}{q:>14.2f}{i:>27.2f}{tail:>24}")
        print(f"{'':<8}checks: " + " ".join(f"{k.name}:{k.status.label}" for k in rep.checks))

    print("\nExperiment B: ten background measurements (crystal sum), baseline = first")
    names = list(SOURCES)
    bg = {s: to_spectrum(rates(s, "bg", "sum"), source=s) for s in names}
    b2 = Baseline.build(mixed_check_source("sum"), bg[names[0]], profile="nai", reference_energies=REFERENCE)
    print(f"{'background of':<14}{'status':<7}{'total rate ratio':>18}  flagged bands")
    for s in names[1:]:
        rep = run_qa(b2, bg[s], mode="background")
        chk = rep.check("background")
        flagged = [d["band"] for d in chk.details if d["status"] != "PASS"]
        warn_by = [k.name for k in rep.checks if k.status.label != "PASS"]
        out["experiment_B"].append({"background_of": s, "status": rep.status.label, "total_rate_ratio": chk.metrics["total_rate_ratio"],
                                    "flagged_bands": flagged, "non_pass_checks": warn_by, "checks": {k.name: k.status.label for k in rep.checks}})
        print(f"{s:<14}{rep.status.label:<7}{chk.metrics['total_rate_ratio']:>18.3f}  bands: {', '.join(flagged[:4]) or '-'} | non-PASS checks: {', '.join(warn_by) or '-'}")
    Path(args.json).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
