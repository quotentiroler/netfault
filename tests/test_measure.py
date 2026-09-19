#!/usr/bin/env python3
#
# The measurement half, tested without an interface.
#
# 'play_record' is the seam: real use hands it a sound device, and here it
# is a function that filters the stimulus through a network whose answer
# is already known in closed form.  So the extraction can be checked
# against the truth to a hundredth of a dB, and the whole path -
# stimulus, capture, level extraction, loopback calibration, fault
# localisation - runs with nothing plugged in.
#
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import deck, record, summary

import netfault
from netfault import measure, mna

HERE = Path(__file__).resolve().parent
FS = 48000
FREQS = list(np.geomspace(40.0, 15000.0, 25))
FACTORS = (0.1, 0.22, 0.47, 2.2, 4.7, 10.0)


def through(src, node="out", gain_db=0.0, noise=0.0, seed=1):
    """A fake interface: filter the stimulus through a real network."""
    rng = np.random.default_rng(seed)

    def play_record(x, fs):
        n = len(x)
        f = np.fft.rfftfreq(n, 1.0 / fs)
        h = np.zeros(len(f), dtype=complex)
        nz = f > 0
        h[nz] = 10.0 ** (mna.solve(src, node, f[nz]) / 20.0)
        y = np.fft.irfft(np.fft.rfft(x) * h, n) * 10.0 ** (gain_db / 20.0)
        return y + rng.normal(0.0, noise, n) if noise else y

    return play_record


def extraction():
    print("level_at  (amplitude of one tone)")
    for dbfs in (-6.0, -20.0, -60.0):
        x = measure.tone(1000.0, 0.2, FS, 10.0 ** (dbfs / 20.0))
        got = measure.level_at(x, 1000.0, FS)
        record(f"recovers {dbfs:.1f} dBFS", abs(got - dbfs) < 0.01,
               f"{got:.4f}")

    x = measure.tone(1000.0, 0.5, FS, 0.5)
    rng = np.random.default_rng(3)
    noisy = x + rng.normal(0.0, 0.01, len(x))
    got = measure.level_at(noisy, 1000.0, FS)
    record("ignores broadband noise 34 dB down",
           abs(got - (-6.0206)) < 0.02, f"{got:.4f}")


def matches_theory():
    print("response  (against the closed-form answer)")
    src = deck("rc.cir")
    got = measure.response(through(src), FREQS, fs=FS)
    want = mna.solve(src, "out", FREQS)
    err = float(np.max(np.abs((got - got.mean()) - (want - want.mean()))))
    record("within 0.01 dB of theory across the band", err < 0.01,
           f"worst {err:.5f} dB")


def calibration():
    print("calibrate  (divide out the interface)")
    src = deck("rc.cir")
    colour = deck("colour.cir")
    raw = measure.response(through(src, gain_db=11.3), FREQS, fs=FS)
    ref = measure.response(through(colour, gain_db=11.3), FREQS, fs=FS)
    want = mna.solve(src, "out", FREQS) - mna.solve(colour, "out", FREQS)
    got = raw - ref
    err = float(np.max(np.abs((got - got.mean()) - (want - want.mean()))))
    record("removes the interface's own response", err < 0.01,
           f"worst {err:.5f} dB")


def end_to_end():
    print("end to end  (measure a faulty board, then name the part)")
    src = deck("ladder.cir")
    sim = mna.runner("out")
    refs = sorted(netfault.components(src))
    cands = netfault.candidates(src, FREQS, sim, refs, FACTORS)

    hits = 0
    trials = [(r, f) for r in refs for f in (0.22, 4.7)]
    for ref, factor in trials:
        faulty = netfault.perturb(src, ref, factor)
        measured = measure.response(through(faulty, gain_db=-7.4, noise=2e-5,
                                            seed=hash(ref) % 999),
                                    FREQS, fs=FS)
        hits += netfault.match(cands, measured)[0][1] == ref
    record("names the part from a measured sweep", hits == len(trials),
           f"{hits}/{len(trials)}")


def budget():
    """The two numbers a bench quotes about itself."""
    print("budget     (what the bench says about itself)")
    src = deck("rc.cir")

    flat = measure.calibrate(through(src, gain_db=6.0), FREQS, fs=FS)
    want = mna.solve(src, "out", FREQS) + 6.0
    record("calibrate reports the loop's own response",
           float(np.max(np.abs(flat - want))) < 0.01,
           f"worst {float(np.max(np.abs(flat - want))):.5f} dB")

    quiet = measure.repeatability(through(src), FREQS, fs=FS)
    record("a bench with no noise repeats exactly", quiet < 1e-9, f"{quiet:.2e} dB")

    loud = [measure.repeatability(through(src, noise=n, seed=7), FREQS, fs=FS)
            for n in (1e-5, 1e-3)]
    record("and a noisier one reads higher, in order",
           quiet <= loud[0] < loud[1], " ".join(f"{v:.5f}" for v in loud))


def main():
    for stage in (extraction, matches_theory, calibration, budget, end_to_end):
        stage()
        print()
    return summary("measure")


if __name__ == "__main__":
    sys.exit(main())
