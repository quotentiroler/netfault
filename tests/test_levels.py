"""A signature is a ladder when the circuit is level dependent."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netfault
from netfault import mna

HERE = Path(__file__).resolve().parent
FREQS = list(np.geomspace(40.0, 15000.0, 21))
FACTORS = (0.22, 0.47, 2.2, 4.7)
LEVELS = (-40.0, -20.0, -6.0)
FAILED = []


def deck(name):
    return (HERE / name).read_text()


def record(name, ok, detail=""):
    print(f"  {name:<52} {'ok' if ok else 'FAIL'}{'  ' + detail if detail else ''}")
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
    def sim(s, f):
        return clipper(s, f, level=-40.0)
    cands = netfault.candidates(src, FREQS, sim, ["Rclip"], FACTORS)
    measured = sim(netfault.perturb(src, "Rclip", 4.7), FREQS)
    spread = max(r[0] for r in netfault.match(cands, measured))
    record("every Rclip candidate sits under the bench floor",
           spread < netfault.FLOOR_DB,
           f"worst residual {spread:.4f} dB, floor {netfault.FLOOR_DB:.2f}")

    v = netfault.explain(cands, measured, noise_db=0.02)
    record("so one sweep refuses to name it", v["verdict"] != "fault",
           "verdict {}, named {}".format(v["verdict"], v["ref"]))


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
           f"{hits}/{len(trials)}")


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
           f"named {top[1]}")


def main():
    for stage in (one_level_is_not_enough, the_ladder_finds_it,
                  each_level_keeps_its_own_gain):
        stage()
        print()
    if FAILED:
        print(f"levels: {len(FAILED)} FAILED")
        for f in FAILED:
            print(f"    {f}")
        return 1
    print("levels: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
