#!/usr/bin/env python3
"""What the circuit will not let anyone measure.

A part the output does not depend on gives a candidate identical to the
nominal curve, and no bench separates two identical curves. Rpd is the
provable case: across an ideal source, so nothing it does reaches a node.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netfault
from netfault import mna

HERE = Path(__file__).resolve().parent
FREQS = [100.0, 1000.0, 10000.0]
REFS = ["Rpd", "R1", "C1"]
FAILED = []


def record(name, ok, detail=""):
    print(f"  {name:<52} {'ok' if ok else 'FAIL'}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def main():
    src = (HERE / "shunt.cir").read_text()
    sim = mna.runner("out")
    cands = netfault.candidates(src, FREQS, sim, REFS, netfault.FACTORS)
    res = netfault.resolution(cands)

    print("a part across an ideal source")
    shunt = [d for d, ref, _f in res if ref == "Rpd"]
    # Seven come back exactly 0.0; OPEN leaves 4.6e-16 of arithmetic.
    record("moves the output by nothing, at every factor",
           len(shunt) == len(netfault.FACTORS) and max(shunt) < 1e-12,
           f"{len(shunt)} factors, worst {max(shunt):.2e} dB")
    record("is dropped even at a floor of zero",
           not any(ref == "Rpd" for ref, _f, _db in netfault.resolvable(cands, 0.0)))
    record("stays dropped at a real bench floor",
           not any(ref == "Rpd" for ref, _f, _db in netfault.resolvable(cands, netfault.FLOOR_DB)))

    print()
    print("what resolvable() must not throw away")
    keep = netfault.resolvable(cands, netfault.FLOOR_DB)
    record("the nominal survives, so a good board can still say so",
           any(ref is None for ref, _f, _db in keep))
    record("parts in the signal path survive",
           {ref for ref, _f, _db in keep if ref} == {"R1", "C1"},
           " ".join(sorted({ref for ref, _f, _db in keep if ref})))

    print()
    print("the ranking it is built on")
    record("resolution is ordered, weakest first",
           all(res[i][0] <= res[i + 1][0] for i in range(len(res) - 1)))
    record("the nominal is not a fault and is not in it",
           all(ref is not None for _d, ref, _f in res))

    print()
    print("two faults the circuit cannot separate")
    # Opening R1 or C1 leaves identical curves, so nothing can choose.
    truth = mna.runner("out")(netfault.perturb(src, "R1", netfault.OPEN), FREQS)
    v = netfault.explain(keep, truth, noise_db=netfault.FLOOR_DB)
    record("are called ambiguous rather than guessed between",
           v["verdict"] == "ambiguous", f"{v['verdict']}, margin {v['margin']:.6f} dB")

    print()
    if FAILED:
        print(f"resolution: {len(FAILED)} FAILED")
        for f in FAILED:
            print(f"    {f}")
        return 1
    print("resolution: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
