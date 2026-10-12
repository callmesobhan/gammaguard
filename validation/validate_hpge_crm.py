"""Validate GammaGuard on real HPGe spectra: 113 ORTEC GMX35P4-70 spectra of pure IAEA reference materials
(RGU-1 uranium ore, RGTh-1 thorium ore, RGK-1 potassium salt), 2013-2022 (Huynh & Truong, Zenodo 23093533).

  H1  calibration / resolution on natural lines. The authors refitted every spectrum's calibration on known
      lines, so reading errors should be ~0: this tests the measurement precision and the fit, not drift.
  H2  unexpected-peak check with ground truth: baseline from the longest RGU-1 spectrum; same-material
      spectra should not warn, Th-ore spectra should (and the library hints should name Th-chain nuclides).
  H3  false-alarm rate of the hot/dead-channel check on real (rebinned) spectra.
  H4  background stability across the four background spectra.

    python validation/validate_hpge_crm.py [--json validation/results/hpge_crm.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gammaguard import Baseline, Status, run_qa
from gammaguard.datasets import HPGE_CRM, load_crm
from gammaguard.datasets.hpge_crm import GRID
from gammaguard.qa.measure import measure_lines
from gammaguard.qa.profiles import HPGE
from gammaguard.resolution import fit_resolution

LINES = {
    "RGU-1": [351.93, 609.31, 1120.29, 1764.49],
    "RGTh-1": [238.63, 583.19, 911.20, 2614.51],
    "RGK-1": [1460.82],
}
TH_CHAIN = ("Tl-208", "Ac-228", "Pb-212", "Bi-212")


def fit_lines(spec, energies):
    """Fit lines near their nominal positions (window +-3 FWHM: the calibration is already refitted)."""
    o = HPGE.fit
    return measure_lines(spec.counts, GRID, energies, lambda e: 2.0, search_fwhm=3.0, step=o.step,
                         roi_fwhm=o.roi_fwhm, multiplet_fwhm=o.multiplet_fwhm)


def h1(crm):
    rows, per_spec = [], []
    for mat, energies in LINES.items():
        for i in crm.indices(material=mat):
            spec = crm.spectrum(i)
            fits = fit_lines(spec, energies)
            e_ok = [e for e, f in zip(energies, fits, strict=True) if f is not None]
            found = [f for f in fits if f is not None]
            for e, f in zip(e_ok, found, strict=True):
                rows.append({"idx": i, "material": mat, "energy": e, "year": int(str(spec.meta["start"])[:4]),
                             "shift_keV": float(GRID(f.centroid) - e), "err_keV": float(0.5 * f.centroid_err),
                             "fwhm_keV": float(0.5 * f.fwhm), "area": f.area, "live_time_s": spec.live_time})
            per_spec.append({"idx": i, "material": mat, "n_expected": len(energies), "n_found": len(found)})
    return rows, per_spec


def h1_summary(rows, per_spec):
    sh = np.array([r["shift_keV"] for r in rows])
    z = np.array([r["shift_keV"] / r["err_keV"] for r in rows])
    out = {
        "n_spectra": len(per_spec), "n_lines_expected": int(sum(p["n_expected"] for p in per_spec)),
        "n_lines_found": len(rows),
        "shift_keV_mean": float(sh.mean()), "shift_keV_median": float(np.median(sh)), "shift_keV_sd": float(sh.std()),
        "shift_keV_p95_abs": float(np.percentile(np.abs(sh), 95)), "shift_keV_max_abs": float(np.abs(sh).max()),
        "fraction_within_0p1_keV": float(np.mean(np.abs(sh) < 0.1)), "median_abs_z": float(np.median(np.abs(z))),
    }
    by_e = {}
    for e in sorted({r["energy"] for r in rows}):
        sel = [r for r in rows if r["energy"] == e]
        by_e[f"{e:.2f}"] = {"n": len(sel), "shift_mean": float(np.mean([r["shift_keV"] for r in sel])),
                            "shift_sd": float(np.std([r["shift_keV"] for r in sel])),
                            "fwhm_median": float(np.median([r["fwhm_keV"] for r in sel]))}
    out["by_energy"] = by_e
    # resolution curve from the median FWHM per energy, and its interpolation at 1332.5 keV
    es = np.array([float(k) for k in by_e])
    fw = np.array([v["fwhm_median"] for v in by_e.values()])
    model = fit_resolution(es, fw)
    out["fwhm_model_a_b"] = [model.a, model.b]
    out["fwhm_at_1332_keV_model"] = float(model.fwhm(1332.49))
    # long-term trend of the FWHM at 609 keV (U-ore) and 1461 keV (K salt): median per year
    trend = {}
    for e in (609.31, 1460.82):
        by_year = {}
        for r in rows:
            if abs(r["energy"] - e) < 0.01:
                by_year.setdefault(r["year"], []).append(r["fwhm_keV"])
        trend[f"{e:.2f}"] = {y: [float(np.median(v)), len(v)] for y, v in sorted(by_year.items())}
    out["fwhm_median_by_year"] = trend
    return out


def h2_h3(crm):
    u = [i for i in crm.indices(material="RGU-1") if crm.meta[i]["tier"] == "gold"]
    # baseline = the highest-statistics acquisition, so that routine (test) spectra have no more
    # structure than the baseline: the unexpected-peak check assumes baseline >= test statistics
    ref = max(u, key=lambda i: crm.counts[i].sum())
    base_spec = crm.spectrum(ref)
    base = Baseline.build(base_spec, profile="hpge", reference_energies=LINES["RGU-1"])
    out = {"baseline_spectrum": crm.meta[ref]["ID"], "baseline_live_time_s": base_spec.live_time, "baseline_total_counts": base_spec.total_counts,
           "baseline_lines": [(ln.label, round(ln.fwhm_keV, 3)) for ln in base.lines], "groups": {}}
    for mat in LINES:
        idx = [i for i in crm.indices(material=mat) if i != ref]
        if mat == "RGU-1":  # same-material test: no more counts than the baseline (see comment above)
            idx = [i for i in idx if crm.counts[i].sum() <= base_spec.total_counts]
        res = {"n": len(idx), "status": {"PASS": 0, "WARN": 0, "FAIL": 0}, "check_non_pass": {}, "unexpected": [],
               "channel_flagged": 0, "channel_flags": 0, "th_hint_fraction": None}
        for i in idx:
            spec = crm.spectrum(i)
            rep = run_qa(base, spec)
            res["status"][rep.status.label] += 1
            for c in rep.checks:
                if c.status >= Status.WARN:
                    res["check_non_pass"][c.name] = res["check_non_pass"].get(c.name, 0) + 1
            up = rep.check("unexpected_peaks")
            res["unexpected"] += [d for d in up.details]
            ch = rep.check("channel_anomalies")
            res["channel_flagged"] += ch.status >= Status.WARN
            res["channel_flags"] += len(ch.details)
        if mat == "RGTh-1":
            hits = [d["candidates"].split(",")[0].strip() for d in res["unexpected"]]
            named = [h for h in hits if h != "none in library"]
            res["th_hint_fraction"] = float(np.mean([any(t in h for t in TH_CHAIN) for h in named])) if named else None
            res["n_hints"] = len(named)
        res["n_unexpected_peaks"] = len(res["unexpected"])
        res["unexpected"] = [{"energy_keV": round(d["energy_keV"], 1), "candidates": d["candidates"]} for d in res["unexpected"][:60]]
        out["groups"][mat] = res
    return out, base


def h4(crm):
    """Background stability. Reference = the longest background, so that the other (shorter) runs have no
    more statistics than the baseline (the unexpected-peak check assumes that)."""
    bk = [crm.background(j) for j in range(len(crm.bkg_meta))]
    ref = max(range(len(bk)), key=lambda j: bk[j].live_time)
    src = crm.spectrum(crm.indices(material="RGU-1")[0])
    b = Baseline.build(src, bk[ref], profile="hpge", reference_energies=LINES["RGU-1"])
    rows = []
    for j in range(len(bk)):
        if j == ref:
            continue
        rep = run_qa(b, bk[j], mode="background")
        chk = rep.check("background")
        rows.append({"background": crm.bkg_meta[j]["file"], "desc": crm.bkg_meta[j]["desc"], "live_h": bk[j].live_time / 3600,
                     "status": rep.status.label, "total_rate_ratio": chk.metrics["total_rate_ratio"],
                     "flagged_bands": [d["band"] for d in chk.details if d["status"] != "PASS"],
                     "non_pass_checks": [c.name for c in rep.checks if c.status >= Status.WARN]})
    return {"reference": crm.bkg_meta[ref]["desc"], "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(Path(__file__).parent / "results" / "hpge_crm.json"))
    args = ap.parse_args()
    crm = load_crm()
    print(f"{HPGE_CRM.title}\n{HPGE_CRM.citation}\n")
    rows, per_spec = h1(crm)
    s1 = h1_summary(rows, per_spec)
    print("H1  natural-line reading errors after the authors' calibration refit")
    print(f"    {s1['n_spectra']} spectra, {s1['n_lines_found']}/{s1['n_lines_expected']} expected lines found")
    print(f"    shift: mean {s1['shift_keV_mean']:+.3f}, sd {s1['shift_keV_sd']:.3f}, 95th pct |shift| {s1['shift_keV_p95_abs']:.3f}, "
          f"max {s1['shift_keV_max_abs']:.3f} keV; {100 * s1['fraction_within_0p1_keV']:.0f} % within 0.1 keV")
    for e, v in s1["by_energy"].items():
        print(f"    {e:>8} keV: n={v['n']:3d}  shift {v['shift_mean']:+.3f} +- {v['shift_sd']:.3f}  FWHM median {v['fwhm_median']:.3f} keV")
    print(f"    FWHM at 1332.5 keV interpolated: {s1['fwhm_at_1332_keV_model']:.3f} keV")
    for e, yrs in s1["fwhm_median_by_year"].items():
        print(f"    FWHM@{e} by year: " + ", ".join(f"{y}: {v[0]:.3f} (n={v[1]})" for y, v in yrs.items()))
    s23, base = h2_h3(crm)
    print(f"\nH2/H3  baseline = {s23['baseline_spectrum']} ({s23['baseline_live_time_s'] / 3600:.0f} h, "
          f"{s23['baseline_total_counts'] / 1e6:.0f} M counts), lines {s23['baseline_lines']}")
    for mat, r in s23["groups"].items():
        print(f"    {mat:<7} n={r['n']:3d}  overall {r['status']}  non-PASS checks {r['check_non_pass']}  "
              f"unexpected peaks {r['n_unexpected_peaks']}  channel-check flags {r['channel_flags']} in {r['channel_flagged']} spectra"
              + (f"  Th-chain hint fraction {r['th_hint_fraction']:.2f} of {r['n_hints']}" if r.get("th_hint_fraction") is not None else ""))
    s4 = h4(crm)
    print(f"\nH4  backgrounds vs the longest one: '{s4['reference']}'")
    for r in s4["rows"]:
        print(f"    {r['desc'][:60]:<60} {r['status']:<5} total rate x{r['total_rate_ratio']:.2f}  bands {r['flagged_bands'][:3]}  checks {r['non_pass_checks']}")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"dataset": HPGE_CRM.key, "H1": s1, "H2_H3": s23, "H4": s4}, indent=1, default=str))
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
