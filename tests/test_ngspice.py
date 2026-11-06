#!/usr/bin/env python3
"""ngspice behind the simulate seam. Skipped when ngspice is not installed."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import deck, record, summary

import netfault
from netfault import mna, ngspice

FREQS = [100.0, 400.0, 1600.0, 6400.0]
LEVELS = (-40.0, -12.0, 6.0)
FACTORS = (0.22, 4.7)


def ac_agrees_with_mna():
    print("ac  (against the numpy solver)")
    src = deck("ladder.cir")
    freqs = list(np.geomspace(20.0, 20000.0, 13))
    d = np.abs(ngspice.ac(src, "out", freqs) - mna.solve(src, "out", freqs))
    record("same curve as mna on the ladder", d.max() < 0.01, f"worst {d.max():.4f} dB")

    cands = netfault.candidates(src, freqs, ngspice.runner("out"), ["R2", "C3"], FACTORS)
    truth = mna.solve(netfault.perturb(src, "C3", 4.7), "out", freqs)
    top = netfault.match(cands, truth)[0]
    record("a dictionary built by ngspice names an mna fault", top[1:] == ("C3", 4.7),
           f"named {top[1]} x{top[2]}")


def tran_is_ac_when_linear():
    print("tran  (a linear deck at low drive)")
    src = deck("ladder.cir")
    d = np.abs(ngspice.tran(src, "out", FREQS, -20.0) - ngspice.ac(src, "out", FREQS))
    record("fundamental gain matches ac", d.max() < 0.01, f"worst {d.max():.4f} dB")


def the_clipper_needs_a_ladder():
    print("tran  (a diode clipper)")
    src = deck("diode.cir")
    sim = ngspice.tran_runner("out")
    low, high = (sim(src, FREQS, level) for level in (-40.0, 6.0))
    squash = low[0] - high[0]
    record("drive compresses it below the corner", squash > 3.0,
           f"{squash:.1f} dB less gain at +6 dBV, {FREQS[0]:g} Hz")

    quiet = netfault.signature(netfault.perturb(src, "Rd", 4.7), FREQS, sim, levels=[-40.0])
    nominal = netfault.signature(src, FREQS, sim, levels=[-40.0])
    gap = netfault.residual(quiet, nominal)
    record("Rd is invisible at low drive", gap < netfault.FLOOR_DB, f"{gap:.4f} dB")

    cands = netfault.candidates(src, FREQS, sim, None, FACTORS, levels=LEVELS)
    hits = 0
    trials = [("Rd", 4.7), ("Rd", 0.22), ("R1", 4.7), ("C1", 0.22)]
    for ref, factor in trials:
        measured = netfault.signature(netfault.perturb(src, ref, factor), FREQS, sim,
                                      levels=LEVELS)
        top = netfault.match(cands, measured)[0]
        hits += top[1:] == (ref, factor)
    record("the ladder names the part, Rd included", hits == len(trials),
           f"{hits}/{len(trials)}")


def failures_are_errors():
    print("what is not an answer")
    floating = "* floating node\nV1 in 0 AC 1\nR1 in out 1k\nC1 out a 1n\nC2 a 0 1n\n.end\n"
    try:
        ngspice.tran(floating, "nowhere", [1000.0], -20.0)
        record("a node ngspice never wrote is an error", ok=False, detail="it returned")
    except ngspice.SpiceError:
        record("a node ngspice never wrote is an error", ok=True, detail="raised")

    two = deck("ladder.cir").replace("RL out 0 100k", "V2 out 0 AC 1")
    try:
        ngspice.drive(two, -20.0)
        record("two AC sources is ambiguous", ok=False, detail="it returned")
    except ValueError:
        record("two AC sources is ambiguous", ok=True, detail="raised")
    driven = ngspice.drive(two, -20.0, source="V2")
    record("unless the source is named", "V2 out 0 DC 0 SIN(" in driven)


def main():
    if not ngspice.available():
        print(f"test-ngspice: SKIPPED - no ngspice on PATH or in ${ngspice.ENV}")
        return 0
    for stage in (ac_agrees_with_mna, tran_is_ac_when_linear, the_clipper_needs_a_ladder,
                  failures_are_errors):
        stage()
        print()
    return summary("test-ngspice")


if __name__ == "__main__":
    sys.exit(main())
