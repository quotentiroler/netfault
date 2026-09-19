#!/usr/bin/env python3
"""A candidate the simulator cannot solve must not cost the dictionary.

Perturbing a part to SHORT or OPEN can leave a nonlinear circuit a real
simulator will not converge on. Measured on a ProCo RAT clone driven
through ngspice: one candidate raised, and it took every other candidate
with it. A dictionary is built once and matched against for the rest of
the board's life, so losing the whole build to one pathological
perturbation is the expensive failure.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netfault
from netfault import mna

HERE = Path(__file__).resolve().parent
FREQS = [100.0, 1000.0, 10000.0]
FAILED = []


def record(name, ok, detail=""):
    print(f"  {name:<52} {'ok' if ok else 'FAIL'}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


class BoomError(RuntimeError):
    """What a simulator raises when it will not converge."""


def sulky(*refuse):
    """A simulate() that raises on the given 1-based call numbers."""
    calls = [0]
    inner = mna.runner("out")

    def simulate(src, freqs, _level=None):
        calls[0] += 1
        if calls[0] in refuse:
            msg = f"will not converge on call {calls[0]}"
            raise BoomError(msg)
        return inner(src, freqs)

    return simulate


def main():
    src = (HERE / "ladder.cir").read_text()
    refs = sorted(netfault.components(src))
    factors = (0.47, 2.2, netfault.OPEN)
    whole = len(refs) * len(factors)

    print("one bad candidate")
    every = netfault.candidates(src, FREQS, sulky(), refs, factors)
    record("a clean run builds every candidate", len(every) == whole + 1,
           f"{len(every) - 1} of {whole}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Call 1 is the nominal, so 3 and 7 are two ordinary candidates.
        survived = netfault.candidates(src, FREQS, sulky(3, 7), refs, factors)
    record("the others survive the ones that will not solve",
           len(survived) - 1 == whole - 2, f"{len(survived) - 1} of {whole}")
    record("the nominal signature is still there",
           any(ref is None for ref, _f, _db in survived))
    record("each one skipped is named", len(caught) == 2,
           "; ".join(str(w.message)[:40] for w in caught))

    print()
    print("what it will not paper over")
    try:
        netfault.candidates(src, FREQS, sulky(1), refs, factors)
        record("a nominal that will not solve is an error", ok=False, detail="it returned")
    except BoomError:
        record("a nominal that will not solve is an error", ok=True, detail="raised")

    print()
    print("the survivors are still usable")
    truth = mna.runner("out")(netfault.perturb(src, "R2", 2.2), FREQS)
    ranked = netfault.match(survived, truth)
    record("a fault still in the dictionary is still found",
           ranked[0][1] == "R2" and abs(ranked[0][2] - 2.2) < 1e-9,
           f"{ranked[0][1]} {ranked[0][2]:g}")

    print()
    if FAILED:
        print(f"unsolvable: {len(FAILED)} FAILED")
        for f in FAILED:
            print(f"    {f}")
        return 1
    print("unsolvable: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
