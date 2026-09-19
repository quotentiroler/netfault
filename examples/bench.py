#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy>=1.20", "sounddevice>=0.4"]
# ///
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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netfault
from netfault import measure, mna

FACTORS = netfault.FACTORS

# Below this the board is passing nothing and no ranking is meaningful.
DEAD_BOARD_DB = -60.0


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
    # Imported here so the library and the tests never need a sound device.
    import sounddevice as sd  # noqa: PLC0415
    if device is not None:
        sd.default.device = device

    def play_record(x, fs):
        y = sd.playrec(x.reshape(-1, 1), samplerate=fs, channels=channels,
                       blocking=True)
        return np.asarray(y)[:, 0]

    return play_record


def bench_devices(args, src):
    """(loop, dut), either a simulated pair or the sound device."""
    if not args.simulate:
        dev = args.device
        if dev and "," in dev:
            dev = tuple(int(v) for v in dev.split(","))
        both = sound_device(dev)
        input("loop the output straight back to the input, then Enter: ")
        return both, both

    board = src
    if args.fault:
        ref, _, factor = args.fault.partition(":")
        board = netfault.perturb(src, ref, float(factor))
        print(f"injected: {ref} x{factor}")
    #
    # The fake bench gets a noise floor too.  A noiseless one is not a
    # preview of anything: every threshold here collapses and it reports
    # confidence no real interface can earn.
    #
    flat = "V1 in 0 AC 1" + chr(10) + "R1 in out 1" + chr(10) + ".end"
    return (simulated_device(flat, "out", gain_db=0.0, noise=3e-5, seed=11),
            simulated_device(board, args.node, gain_db=-3.0, noise=3e-5, seed=12))


def report(v, parts, noise, ranked):
    """What the verdict means, for somebody holding an iron."""
    if v["verdict"] == "unexplained":
        print(f"  NOTHING HERE EXPLAINS THIS ({v['residual']:.4f} dB residual"
              f" against a {noise:.4f} dB bench).")
        print("  The deck does not describe the rig.  Usual causes: the"
              " interface's own")
        print("  source and load impedance missing from the netlist, or"
              " stray cable")
        print("  capacitance.  Fix the deck before reading anything above.")
        return

    if v["verdict"] == "ambiguous":
        nxt = ranked[1]
        if v["ref"] is None:
            print(f"  CLOSE TO NOMINAL, but {nxt[1]} at {nxt[2]:g}x fits within"
                  f" {v['margin']:.4f} dB of it.")
            print("  This bench cannot tell those two apart.")
        else:
            print(f"  AMBIGUOUS: {v['ref']} at {v['factor']:g}x fits, but only"
                  f" {v['margin']:.4f} dB better than"
                  f" {nxt[1] or 'nominal'} at {nxt[2]:g}x.")
        print("  Average longer (--seconds) or add points (--points).")
        return

    if v["verdict"] == "nominal":
        print(f"  the board matches the netlist ({v['residual']:.4f} dB)")
        return

    d = netfault.describe(v["factor"])
    if d in ("OPEN", "SHORT"):
        print(f"  {v['ref']} is {d}  ({parts[v['ref']]['raw']} - check the"
              " joint, check for a bridge)")
    else:
        print(f"  {v['ref']} at {v['factor']:g}x nominal"
              f" ({parts[v['ref']]['value']:.3g} ->"
              f" {parts[v['ref']]['value'] * v['factor']:.3g})")
    print(f"  {v['residual']:.4f} dB residual, {v['margin']:.4f} dB clear of"
          " the next answer")


def printable(text):
    """Device names carry anything the driver felt like, consoles do not."""
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, "replace").decode(encoding)


def list_devices():
    """Every device and its id, so --device has something to name."""
    import sounddevice as sd  # noqa: PLC0415

    default_in, default_out = sd.default.device
    for i, d in enumerate(sd.query_devices()):
        legs = []
        if d["max_input_channels"]:
            legs.append(f"in x{d['max_input_channels']}")
        if d["max_output_channels"]:
            legs.append(f"out x{d['max_output_channels']}")
        mark = "".join(("<" if i == default_in else " ", ">" if i == default_out else " "))
        print(f"  {i:>3} {mark} {printable(d['name'])[:44]:<44} {', '.join(legs)}")
    print()
    print("  < is the current input, > the current output.")
    print("  Pass one id for both, or 'in,out' for a pair:  --device 1,3")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("netlist", nargs="?", help="the deck the board claims to be")
    ap.add_argument("--list-devices", action="store_true",
                    help="print every device and its id, then stop")
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
    ap.add_argument("--exclude", default="",
                    help="parts that are not on the board and so cannot be "
                         "mis-fitted: the interface's own Rsrc, Rin, ...")
    ap.add_argument("--top", type=int, default=5)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.list_devices:
        list_devices()
        return 0
    if not args.netlist:
        sys.exit("which netlist? pass one, or --list-devices to see the hardware")

    src = Path(args.netlist).read_text()
    freqs = list(np.geomspace(args.lo, args.hi, args.points))
    amp = 10.0 ** (args.level / 20.0)
    kw = {"fs": args.fs, "seconds": args.seconds, "amplitude": amp}

    parts = netfault.components(src)
    if not parts:
        sys.exit(f"no R/C/L with plain values in {args.netlist}")
    print(f"{Path(args.netlist).name}: {len(parts)} parts, "
          f"{len(freqs)} points from {args.lo:g} to {args.hi:g} Hz")

    loop, dut = bench_devices(args, src)

    noise = measure.repeatability(loop, freqs, **kw)
    print(f"bench repeatability: {noise:.4f} dB RMS")
    ref_db = measure.calibrate(loop, freqs, **kw)
    if not args.simulate:
        input("now put the board in the loop, then Enter: ")
    meas = measure.response(dut, freqs, **kw) - ref_db

    sim = mna.runner(args.node)
    skip = {r.strip() for r in args.exclude.split(",") if r.strip()}
    refs = [r for r in sorted(parts) if r not in skip]
    if skip:
        print("not on the board, so not candidates: {}".format(", ".join(sorted(skip))))
    cands = netfault.candidates(src, freqs, sim, refs, FACTORS)
    #
    # A board passing nothing is the commonest complaint of all, and the
    # ranking has nothing useful to say about it: every candidate is being
    # compared against a noise floor.  Answer it before trying.
    #
    if float(np.mean(meas)) < DEAD_BOARD_DB:
        print()
        print(f"  THE BOARD IS PASSING ALMOST NOTHING ({float(np.mean(meas)):.1f} dB mean).")
        print("  That is an open somewhere in the signal path, or no board"
              " in the loop.")
        print("  Check continuity end to end before measuring anything.")
        return 0

    usable = netfault.resolvable(cands, max(noise, 1e-4))
    dropped = len(cands) - len(usable)
    if dropped:
        print(f"  {dropped} of {len(cands) - 1} candidates are below this bench's"
              " resolution and were not considered")
    v = netfault.explain(usable, meas, noise_db=max(noise, 1e-4))
    ranked = v["ranked"]

    print(f"\n  {'part':<10} {'factor':<8} residual dB")
    for resid, ref, factor in ranked[:args.top]:
        shown = "-" if ref is None else netfault.describe(factor)
        print(f"  {ref or '(nominal)':<10} {shown:<8} {resid:.4f}")

    print()
    report(v, parts, noise, ranked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
