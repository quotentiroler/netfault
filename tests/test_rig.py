#!/usr/bin/env python3
"""What the deck does not have: stray capacitance, solder bridges, values off the grid."""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import deck, record, summary

import netfault
from netfault import mna

FREQS = list(np.geomspace(20.0, 20000.0, 25))
NOISE = 0.02


def diagnose(src, board, **kw):
    sim = mna.runner("out")
    return netfault.diagnose(src, FREQS, sim, sim(board, FREQS), noise_db=NOISE, **kw)


def said(v):
    return "{} {} {}".format(v["verdict"], v["ref"], netfault.describe(v["factor"], v["ref"]))


def stray_is_named():
    print("stray capacitance")
    src = deck("tap.cir")
    board = netfault.attach(src, "Ccable out 0 1.5n")

    v = diagnose(src, board, rig=False)
    record("without the rig it is unexplained, as before", v["verdict"] == "unexplained", said(v))

    v = diagnose(src, board)
    record("with it, 1.5 nF at out is the answer", v["verdict"] == "fault"
           and v["ref"] == "+C(out)" and abs(v["factor"] / 1.5e-9 - 1.0) < 0.01, said(v))

    src = deck("ladder.cir")
    v = diagnose(src, netfault.attach(src, "Ccable a 0 2.2n"), rig=False)
    named = {v["ref"]: v["factor"], **dict(v["twins"])}
    record("on a node that has a capacitor, it is that capacitor larger",
           abs(named.get("C1", 0.0) - 1.22) < 0.01 and "+C(a)" in named, str(named))


def off_the_grid():
    print("values the grid does not hold")
    src = deck("ladder.cir")
    for ref, value, board in (
            ("+C(out)", 1.5e-9, netfault.attach(src, "Ccable out 0 1.5n")),
            ("+R(b,0)", 330.0, netfault.attach(src, "Rbridge b 0 330")),
            ("R3", 1.6, netfault.perturb(src, "R3", 1.6)),
            ("R2", 0.33, netfault.perturb(src, "R2", 0.33))):
        v = diagnose(src, board)
        named = {v["ref"]: v["factor"], **dict(v["twins"])}
        found = named.get(ref)
        record(f"{ref} {netfault.describe(value, ref)} is found at its own value",
               v["verdict"] == "fault" and found is not None and abs(found / value - 1.0) < 0.01,
               said(v))


def bridges():
    print("solder bridges")
    src = deck("ladder.cir")
    v = diagnose(src, netfault.attach(src, "Rbridge a out 1"))
    record("a bridge between unconnected nodes", v["ref"] == "+R(a,out)", said(v))
    v = diagnose(src, netfault.attach(src, "Rbridge b 0 10"))
    record("a bridge across a capacitor is not that capacitor shorted",
           v["ref"] == "+R(b,0)", said(v))
    record("a bridge across a resistor is that resistor, so is not a candidate",
           not any(r[1] == "+R(a,b)" for r in netfault.additions(
               src, FREQS, mna.runner("out"), stray=())), "R2 joins a and b")


def no_new_doubt():
    print("what the rig must not break")
    src = deck("ladder.cir")
    v = diagnose(src, src)
    record("a healthy board is still nominal", v["verdict"] == "nominal", said(v))

    rng = np.random.default_rng(3)
    sim = mna.runner("out")
    noisy = sim(src, FREQS) + rng.normal(0.0, NOISE, len(FREQS))
    v = netfault.diagnose(src, FREQS, sim, noisy, noise_db=NOISE)
    record("and still nominal through the bench's own noise", v["verdict"] == "nominal",
           said(v))

    v = diagnose(src, netfault.perturb(src, "C2", 4.7))
    record("a real part fault is still that part", v["ref"] == "C2", said(v))

    v = diagnose(src, netfault.perturb(src, "R1", 0.7))
    record("R1 low and a leak from a to ground look the same, and it says so",
           v["verdict"] == "ambiguous" and {v["ref"], v["ranked"][1][1]} >= {"+R(a,0)"},
           "{} vs {}".format(v["ref"], v["ranked"][1][1]))


def capacitors_short_upwards():
    print("describe")
    record("a capacitor scaled up is a short", netfault.describe(netfault.OPEN, "C1") == "SHORT")
    record("a resistor scaled up is an open", netfault.describe(netfault.OPEN, "R1") == "OPEN")
    record("an added part reads as its value", netfault.describe(2.2e-9, "+C(a)") == "2.2nF")


def main():
    warnings.simplefilter("ignore", netfault.UnsolvableCandidate)
    for stage in (stray_is_named, off_the_grid, bridges, no_new_doubt,
                  capacitors_short_upwards):
        stage()
        print()
    return summary("test-rig")


if __name__ == "__main__":
    sys.exit(main())
