"""A signature is a ladder when the circuit is level dependent."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import netfault
from netfault import mna

HERE = os.path.dirname(os.path.abspath(__file__))
FREQS = list(np.geomspace(40.0, 15000.0, 21))
FACTORS = (0.22, 0.47, 2.2, 4.7)
LEVELS = (-40.0, -20.0, -6.0)
FAILED = []


def deck(name):
    return open(os.path.join(HERE, name)).read()


def record(name, ok, detail=""):
    print("  %-52s %s%s" % (name, "ok" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        FAILED.append(name)


def clipper(src, freqs, level=None):
    """Linear below the knee; above it Rclip sets how hard it compresses.

    Stands in for a real nonlinear solve.  What matters for the ranking
    is that one component is invisible until the drive is high enough,
    which is the property a single sweep cannot see.
    """
    db = mna.solve(src, "out", freqs)
    if level is None or level <= -30.0:
        return db
    rclip = netfault.components(src)["Rclip"]["value"]
    knee = (level + 30.0) / 24.0
    return db - knee * 20.0 * np.log10(1.0 + 1e8 / rclip)


def one_level_is_not_enough():
    print("a single low-drive sweep cannot see the clipper")
    src = deck("clipper.cir")
    sim = lambda s, f: clipper(s, f, level=-40.0)
    cands = netfault.candidates(src, FREQS, sim, ["Rclip"], FACTORS)
    measured = sim(netfault.perturb(src, "Rclip", 4.7), FREQS)
    spread = max(r[0] for r in netfault.match(cands, measured))
    record("every Rclip candidate sits under the bench floor",
           spread < netfault.FLOOR_DB,
           "worst residual %.4f dB, floor %.2f" % (spread, netfault.FLOOR_DB))

    v = netfault.explain(cands, measured, noise_db=0.02)
    record("so one sweep refuses to name it", v["verdict"] != "fault",
           "verdict %s, named %s" % (v["verdict"], v["ref"]))


def the_ladder_finds_it():
    print("the ladder does")
    src = deck("clipper.cir")
    refs = sorted(netfault.components(src))
    cands = netfault.candidates(src, FREQS, clipper, refs, FACTORS, levels=LEVELS)
    record("a candidate carries one row per level",
           np.asarray(cands[0][2]).shape == (len(LEVELS), len(FREQS)),
           str(np.asarray(cands[0][2]).shape))

    hits = 0
    trials = [(r, f) for r in refs for f in (0.22, 4.7)]
    for ref, factor in trials:
        measured = netfault.signature(netfault.perturb(src, ref, factor), FREQS,
                                      clipper, levels=LEVELS)
        top = netfault.match(cands, measured)[0]
        hits += top[1] == ref
    record("names the part, clipper included", hits == len(trials),
           "%d/%d" % (hits, len(trials)))


def each_level_keeps_its_own_gain():
    """A measured ladder has a different offset per level, not one offset."""
    print("per-level alignment")
    src = deck("clipper.cir")
    refs = sorted(netfault.components(src))
    cands = netfault.candidates(src, FREQS, clipper, refs, FACTORS, levels=LEVELS)
    truth = netfault.signature(netfault.perturb(src, "R2", 4.7), FREQS,
                               clipper, levels=LEVELS)
    skewed = np.asarray(truth) + np.array([[3.0], [-7.5], [11.0]])
    top = netfault.match(cands, skewed)[0]
    record("survives a different offset on every rung", top[1] == "R2",
           "named %s" % top[1])


def main():
    for stage in (one_level_is_not_enough, the_ladder_finds_it,
                  each_level_keeps_its_own_gain):
        stage()
        print()
    if FAILED:
        print("levels: %d FAILED" % len(FAILED))
        for f in FAILED:
            print("    %s" % f)
        return 1
    print("levels: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
