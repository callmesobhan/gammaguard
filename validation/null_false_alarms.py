"""False-alarm rate on healthy simulated detectors: fresh baseline and fresh test spectrum per trial.

    python validation/null_false_alarms.py [--trials 400]
"""

import argparse
from collections import Counter

import numpy as np

from gammaguard import Baseline, Status, run_qa
from gammaguard.simulate import background_scenario, simulate, source_scenario


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--seed", type=int, default=70000)
    args = ap.parse_args()
    for det in ("hpge", "nai"):
        for mode in ("source", "background"):
            alarms: Counter = Counter()
            for k in range(args.trials):
                rs = np.random.SeedSequence(args.seed + k).spawn(3)
                b = Baseline.build(simulate(source_scenario(1800, det), np.random.default_rng(rs[0])),
                                   simulate(background_scenario(4 * 3600, det), np.random.default_rng(rs[1])), profile=det)
                sc = source_scenario(600, det) if mode == "source" else background_scenario(3600, det)
                rep = run_qa(b, simulate(sc, np.random.default_rng(rs[2])), mode=mode)
                if rep.status >= Status.WARN:
                    alarms["ANY"] += 1
                    alarms.update(c.name for c in rep.checks if c.status >= Status.WARN)
            print(f"{det:<5} {mode:<11} {args.trials} healthy trials: " + (str(dict(alarms)) if alarms else "no WARN or FAIL"))


if __name__ == "__main__":
    main()
