#!/usr/bin/env python3
#
# Diagnose a board on the desk.
#
#   1. loop the interface's output straight back to its input, calibrate
#   2. put the board in the loop, measure
#   3. rank every single-component fault by how well it explains the two
#
# --simulate runs the whole thing against the netlist instead of a sound
# card, so the install and the netlist can be checked before any cable is
# plugged in.  --fault injects a known one, which is how you find out what
# the rig would have said if a part really were wrong.
#
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import netfault
from netfault import measure, mna

FACTORS = (0.1, 0.22, 0.33, 0.47, 0.68, 1.5, 2.2, 3.3, 4.7, 10.0)


def simulated_device(src, node, gain_db=-3.0, noise=0.0, seed=0):
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


def sound_device(device=None, channels=1):
    import sounddevice as sd
    if device is not None:
        sd.default.device = device

    def play_record(x, fs):
        y = sd.playrec(x.reshape(-1, 1), samplerate=fs, channels=channels,
                       blocking=True)
        return np.asarray(y)[:, 0]

    return play_record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("netlist", help="the deck the board claims to be")
    ap.add_argument("--node", default="out", help="output node name")
    ap.add_argument("--fs", type=int, default=48000)
    ap.add_argument("--points", type=int, default=25)
    ap.add_argument("--lo", type=float, default=40.0)
    ap.add_argument("--hi", type=float, default=15000.0)
    ap.add_argument("--seconds", type=float, default=0.25,
                    help="per tone; raise it to average down noise")
    ap.add_argument("--level", type=float, default=-12.0,
                    help="stimulus, dBFS - leave headroom, do not clip")
    ap.add_argument("--device", default=None,
                    help="sounddevice id, 'in,out' - omit for the default")
    ap.add_argument("--simulate", action="store_true",
                    help="no hardware: run against the netlist itself")
    ap.add_argument("--fault", default=None, metavar="REF:FACTOR",
                    help="with --simulate, inject a known fault")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    src = open(args.netlist).read()
    freqs = list(np.geomspace(args.lo, args.hi, args.points))
    amp = 10.0 ** (args.level / 20.0)
    kw = dict(fs=args.fs, seconds=args.seconds, amplitude=amp)

    parts = netfault.components(src)
    if not parts:
        sys.exit("no R/C/L with plain values in %s" % args.netlist)
    print("%s: %d parts, %d points from %g to %g Hz"
          % (os.path.basename(args.netlist), len(parts), len(freqs),
             args.lo, args.hi))

    if args.simulate:
        board = src
        if args.fault:
            ref, _, factor = args.fault.partition(":")
            board = netfault.perturb(src, ref, float(factor))
            print("injected: %s x%s" % (ref, factor))
        loop = simulated_device("V1 in 0 AC 1\nR1 in out 1\n.end", "out")
        dut = simulated_device(board, args.node, gain_db=-3.0, noise=3e-5)
    else:
        dev = args.device
        if dev and "," in dev:
            dev = tuple(int(v) for v in dev.split(","))
        loop = dut = sound_device(dev)
        input("loop the output straight back to the input, then Enter: ")

    ref_db = measure.calibrate(loop, freqs, **kw)
    if not args.simulate:
        input("now put the board in the loop, then Enter: ")
    meas = measure.response(dut, freqs, **kw) - ref_db

    sim = mna.runner(args.node)
    cands = netfault.candidates(src, freqs, sim, sorted(parts), FACTORS)
    ranked = netfault.match(cands, meas)

    print("\n  %-10s %-8s %s" % ("part", "factor", "residual dB"))
    for resid, ref, factor in ranked[:args.top]:
        print("  %-10s %-8s %.4f"
              % (ref or "(nominal)", "-" if ref is None else "x%g" % factor,
                 resid))

    best, runner = ranked[0], ranked[1]
    print()
    if best[1] is None:
        print("  the board matches the netlist (%.4f dB)" % best[0])
    else:
        margin = runner[0] - best[0]
        print("  best explanation: %s at %g x nominal (%.3g -> %.3g)"
              % (best[1], best[2], parts[best[1]]["value"],
                 parts[best[1]]["value"] * best[2]))
        print("  %.4f dB residual, %.4f dB clear of the next answer" %
              (best[0], margin))
        if margin < 0.02:
            print("  MARGIN IS THIN - average longer (--seconds) before"
                  " believing this")
    return 0


if __name__ == "__main__":
    sys.exit(main())
