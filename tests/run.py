#!/usr/bin/env python3
#
# Everything here runs on python and numpy alone.  No simulator, no
# subprocess, no hardware: the network under test is linear, so its
# answer is a solve rather than a program, and a fault is INJECTED rather
# than soldered.  Scale one component by a known factor, hand the result
# to the localiser as though it came off a board, and it has to name the
# part it was never told about.
#
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netfault
from netfault import mna

HERE = Path(__file__).resolve().parent
FREQS = list(np.geomspace(20.0, 20000.0, 25))
FACTORS = (0.1, 0.22, 0.47, 2.2, 4.7, 10.0)
FAILED = []


def deck(name):
    return (HERE / name).read_text()


def record(name, ok, detail=""):
    print(f"  {name:<48} {'ok' if ok else 'FAIL'}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def solver_is_right():
    """The solver against a first-order low-pass solved by hand."""
    print("mna  (against closed form)")
    src = deck("rc.cir")
    fc = 1000.0
    db = mna.solve(src, "out", [fc / 100, fc, fc * 10])
    record("passband is 0 dB", abs(db[0]) < 0.01, f"{db[0]:.4f} dB")
    record("-3.0103 dB at the corner", abs(db[1] + 3.0103) < 0.01,
           f"{db[1]:.4f} dB")
    record("-20 dB/decade above it", abs(db[2] + 20.04) < 0.1,
           f"{db[2]:.4f} dB")


def values():
    print("parse_value")
    for text, want in (("100k", 1e5), ("4.7k", 4700.0), ("1n", 1e-9),
                       ("10u", 1e-5), ("1meg", 1e6), ("100p", 1e-10),
                       ("47", 47.0), ("2.2u", 2.2e-6)):
        got = netfault.parse_value(text)
        record(f"{text:<8} -> {want:g}", abs(got - want) <= abs(want) * 1e-9,
               f"got {got:g}")


def parsing():
    print("components / perturb")
    src = deck("ladder.cir")
    c = netfault.components(src)
    record("finds every R/C", set(c) == {"R1", "R2", "R3", "C1", "C2", "C3",
                                         "RL"}, str(sorted(c)))
    record("R2 is 22k", abs(c["R2"]["value"] - 22e3) < 1e-6)
    out = netfault.perturb(src, "R2", 10.0)
    diff = [i for i, (x, y) in enumerate(zip(src.splitlines(),
                                             out.splitlines())) if x != y]
    record("changes exactly one line", len(diff) == 1, f"changed {len(diff)}")
    record("R2 is now 220k",
           abs(netfault.components(out)["R2"]["value"] - 220e3) < 1e-6)


def finds_the_fault():
    print("localise  (inject a fault, then find it)")
    src = deck("ladder.cir")
    sim = mna.runner("out")
    refs = sorted(netfault.components(src))
    cands = netfault.candidates(src, FREQS, sim, refs, FACTORS)
    record("library covers every part at every factor",
           len(cands) == len(refs) * len(FACTORS) + 1, f"{len(cands)}")

    hits = 0
    trials = [(r, f) for r in refs for f in (0.22, 4.7)]
    for ref, factor in trials:
        measured = sim(netfault.perturb(src, ref, factor), FREQS)
        top = netfault.match(cands, measured)[0]
        hits += top[1] == ref and abs(top[2] - factor) < 1e-9
    record("names the part and the factor, every time",
           hits == len(trials), f"{hits}/{len(trials)}")

    nominal = netfault.match(cands, sim(src, FREQS))[0]
    record("a correct board is called correct", nominal[1] is None,
           f"named {nominal[1]}")


def realism():
    """A measured leg has a gain offset and noise; a solve does not."""
    print("realism  (what a real measurement does to the ranking)")
    src = deck("ladder.cir")
    sim = mna.runner("out")
    refs = sorted(netfault.components(src))
    cands = netfault.candidates(src, FREQS, sim, refs, FACTORS)
    truth = {r: sim(netfault.perturb(src, r, 4.7), FREQS) for r in refs}

    for offset in (6.0, -13.7):
        ok = all(netfault.match(cands, truth[r] + offset)[0][1] == r
                 for r in refs)
        record(f"survives a {offset:+.1f} dB gain offset", ok)

    rng = np.random.default_rng(20260914)
    print(f"    top-1 over {len(refs)} parts, 20 draws each:")
    rate = {}
    for sigma in (0.02, 0.05, 0.10, 0.20):
        hits = 0
        for r in refs:
            for _ in range(20):
                m = truth[r] + rng.normal(0.0, sigma, len(FREQS)) + 6.0
                hits += netfault.match(cands, m)[0][1] == r
        rate[sigma] = 100.0 * hits / (20 * len(refs))
        print(f"      {sigma:.2f} dB RMS -> {rate[sigma]:3.0f}%")
    record("clean measurement is unambiguous", rate[0.02] == 100.0,
           f"{rate[0.02]:.0f}%")
    record("0.05 dB RMS is a workable bench budget", rate[0.05] >= 90.0,
           f"{rate[0.05]:.0f}%")


def refuses_to_guess():
    """The half the first round of goals left out.

    Every criterion up to here was 'given a fault, find it'.  None of them
    said 'given no fault and a deck that does not quite describe the rig,
    do not invent one', and that is the failure that matters: a wrong
    answer delivered with a good margin is worse than no answer.
    """
    print("explain  (when nothing here is the answer)")
    src = deck("ladder.cir")
    sim = mna.runner("out")
    refs = sorted(netfault.components(src))
    cands = netfault.candidates(src, FREQS, sim, refs, FACTORS)
    noise = 0.02

    v = netfault.explain(cands, sim(src, FREQS), noise_db=noise)
    record("a correct board on a correct deck is nominal",
           v["verdict"] == "nominal", v["verdict"])

    faulty = netfault.perturb(src, "C2", 4.7)
    v = netfault.explain(cands, sim(faulty, FREQS), noise_db=noise)
    record("a real fault is still named",
           v["verdict"] == "fault" and v["ref"] == "C2",
           "{} {}".format(v["verdict"], v["ref"]))

    # the rig has something the deck never heard of - stray cable capacitance
    stray = src.replace(".end", "Ccable out 0 2.2n" + chr(10) + ".end")
    v = netfault.explain(cands, sim(stray, FREQS), noise_db=noise)
    record("an undeclared part is not blamed on a declared one",
           v["verdict"] == "unexplained",
           "{} -> {}".format(v["verdict"], v["ref"]))

    v = netfault.explain(cands, sim(src, FREQS), noise_db=noise,
                         margin_db=10.0)
    record("a thin margin is called ambiguous", v["verdict"] == "ambiguous",
           v["verdict"])


def main():
    for stage in (solver_is_right, values, parsing, finds_the_fault,
                  realism, refuses_to_guess):
        stage()
        print()
    if FAILED:
        print(f"netfault: {len(FAILED)} FAILED")
        for f in FAILED:
            print(f"    {f}")
        return 1
    print("netfault: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
