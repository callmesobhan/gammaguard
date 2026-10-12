"""Command-line interface: ``gammaguard {simulate,baseline,check,benchmark,demo}``.

``check`` exits 0 (pass), 1 (warn) or 2 (fail), so it can gate a cron job or CI step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .io import read_spectrum, write_spectrum
from .qa import Baseline, run_qa
from .simulate import FAULTS, background_scenario, inject, parse_fault, simulate, source_scenario


def _cmd_simulate(args: argparse.Namespace) -> int:
    live = args.live_time or (600.0 if args.kind == "source" else 3600.0)
    sc = source_scenario(live, args.detector) if args.kind == "source" else background_scenario(live, args.detector)
    sc = inject(sc, *(parse_fault(f) for f in args.fault))
    spec = simulate(sc, args.seed)
    write_spectrum(spec, args.out)
    print(f"wrote {args.out}: {spec.total_counts} counts, {live:g} s live"
          + (f", faults: {', '.join(sc.faults)}" if sc.faults else ""))
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    background = read_spectrum(args.background) if args.background else None
    base = Baseline.build(read_spectrum(args.source), background, profile=args.profile)
    base.save(args.out)
    print(f"wrote {args.out} (profile {base.profile})")
    for ln in base.lines:
        pct = 100 * ln.fwhm_keV / ln.energy
        print(f"  {ln.label:<14} FWHM {ln.fwhm_keV:.3f} +- {ln.fwhm_err_keV:.3f} keV ({pct:.2f} %)")
    print(f"  calibration (keV): {', '.join(f'{c:.6g}' for c in base.calibration.coeffs)}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    report = run_qa(Baseline.load(args.baseline), read_spectrum(args.spectrum), mode=args.mode)
    text = {"text": report.to_text, "json": report.to_json, "md": report.to_markdown}[args.format]()
    if args.output:
        Path(args.output).write_text(text if text.endswith("\n") else text + "\n")
    else:
        print(text)
    return report.exit_code


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from .bench import format_results, run_benchmark

    print(f"detector: {args.detector}")
    print(format_results(run_benchmark(args.trials, args.seed, detector=args.detector)))
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    from .datasets import DATASETS, fetch

    keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for key in keys:
        ds = DATASETS[key]
        print(f"{ds.key}: {ds.title}\n  detector: {ds.detector}\n  licence:  {ds.license}\n  note:     {ds.note}")
        print(f"  extracted to {fetch(ds)}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    det = args.detector
    rng = np.random.default_rng(args.seed)
    good_source = simulate(source_scenario(1800.0, det), rng)
    good_background = simulate(background_scenario(4 * 3600.0, det), rng)
    base = Baseline.build(good_source, good_background, profile=det)
    hot, spur = (2200, 800.0) if det == "hpge" else (650, 2200.0)
    runs = [
        ("healthy detector", "source", []),
        ("gain drift " + ("+0.05 %" if det == "hpge" else "+1 %"), "source",
         ["gain_drift:" + ("0.0005" if det == "hpge" else "0.01")]),
        ("resolution x1.20", "source", ["resolution_loss:1.2"]),
        (f"unexpected {spur:g} keV line", "source", [f"spurious_peak:{spur:g}:{1.0 if det == 'hpge' else 1.5}"]),
        ("hot channel", "source", [f"hot_channel:{hot}:{0.5 if det == 'hpge' else 2.0}"]),
        ("background x2.5", "background", ["background_scale:2.5"]),
    ]
    for title, mode, faults in runs:
        sc = source_scenario(600.0, det) if mode == "source" else background_scenario(3600.0, det)
        spec = simulate(inject(sc, *(parse_fault(f) for f in faults)), rng)
        report = run_qa(base, spec, mode=mode)
        print(f"\n=== {title} ===")
        print(report.to_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gammaguard", description=__doc__.split("\n")[0])
    p.add_argument("--version", action="version", version=f"gammaguard {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="write a simulated spectrum (optionally with injected faults)")
    s.add_argument("--kind", choices=("source", "background"), default="source")
    s.add_argument("--detector", choices=("hpge", "nai"), default="hpge")
    s.add_argument("--live-time", type=float, help="seconds (default: 600 source, 3600 background)")
    s.add_argument("--fault", action="append", default=[], metavar="NAME[:ARG...]",
                   help=f"inject a fault; one of {', '.join(sorted(FAULTS))} (repeatable)")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--out", required=True)
    s.set_defaults(func=_cmd_simulate)

    b = sub.add_parser("baseline", help="build a baseline from known-good measurements")
    b.add_argument("--source", required=True, help="check-source spectrum from a healthy detector")
    b.add_argument("--background", help="background spectrum from a healthy detector")
    b.add_argument("--profile", choices=("hpge", "nai"), default="hpge",
                   help="detector type (peak model, reference lines)")
    b.add_argument("--out", required=True)
    b.set_defaults(func=_cmd_baseline)

    c = sub.add_parser("check", help="grade a spectrum against a baseline (exit 0/1/2 = pass/warn/fail)")
    c.add_argument("--baseline", required=True)
    c.add_argument("--mode", choices=("source", "background"), default="source")
    c.add_argument("--format", choices=("text", "json", "md"), default="text")
    c.add_argument("--output", help="write the report here instead of stdout")
    c.add_argument("spectrum")
    c.set_defaults(func=_cmd_check)

    k = sub.add_parser("benchmark", help="Monte Carlo detection / false-alarm table")
    k.add_argument("--trials", type=int, default=20)
    k.add_argument("--seed", type=int, default=2026)
    k.add_argument("--detector", choices=("hpge", "nai"), default="hpge")
    k.set_defaults(func=_cmd_benchmark)

    d = sub.add_parser("demo", help="end-to-end tour on simulated data")
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--detector", choices=("hpge", "nai"), default="hpge")
    d.set_defaults(func=_cmd_demo)

    f = sub.add_parser("fetch-data", help="download the open validation datasets (not bundled; own licences)")
    f.add_argument("dataset", choices=("eth_rll", "hpge_crm", "all"))
    f.set_defaults(func=_cmd_fetch)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"gammaguard: error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
