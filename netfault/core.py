#
# Which component is wrong?
#
# Measuring that a build and its schematic disagree is the easy half, and
# not the useful one: a curve sagging above 2 kHz is a symptom, and what
# somebody holding a soldering iron needs is a reference designator.
#
# So the question is turned around.  Rather than inverting the
# measurement, every single-component fault that could explain it is
# simulated and the closest one wins.  That is brute force and completely
# robust: no derivative, no assumption that the response is linear in the
# part values, and no way to be fooled by a topology it was not told
# about, because every candidate IS the topology.
#
# One component at a time is the honest limit.  Two parts wrong at once
# is a much larger search and a much weaker claim, and a board with two
# faults usually announces itself anyway.
#
import re

import numpy as np

from .values import format_value, parse_value

_DEVICE = re.compile(r"^([RCL]\w*)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)\s*$", re.MULTILINE)


def components(src):
    """Every R/C/L with a plain numeric value, as {ref: {nodes, value}}.

    A device whose value is an expression is deliberately not a candidate:
    it is a modelling parameter rather than a part anybody can fit
    backwards.
    """
    out = {}
    for m in _DEVICE.finditer(src):
        try:
            value = parse_value(m.group(7))
        except ValueError:
            continue
        out[m.group(1)] = {"nodes": (m.group(3), m.group(5)), "value": value,
                           "raw": m.group(7)}
    return out


def perturb(src, ref, factor):
    """The same deck with one component scaled, and nothing else touched."""
    seen = []

    def swap(m):
        if m.group(1) != ref:
            return m.group(0)
        seen.append(ref)
        return "".join(m.group(1, 2, 3, 4, 5, 6)) + \
            format_value(parse_value(m.group(7)) * factor)

    out = _DEVICE.sub(swap, src)
    if len(seen) != 1:
        msg = f"{ref} matched {len(seen)} devices, wanted 1"
        raise ValueError(msg)
    return out


def signature(src, freqs, simulate, levels=None):
    """What a board looks like: one dB curve, or one per drive level.

    A linear network is the same network at every level, so one curve
    says everything about it.  A circuit with a clipper in it is not:
    the part that sets where it folds does nothing at all until the
    drive reaches it, so a single sweep cannot tell a wrong one from a
    right one.  Ask at several levels and it can.

    'simulate' is called as simulate(src, freqs) without levels, and as
    simulate(src, freqs, level) with them.  Returns a 1-D array in the
    first case and a (level, freq) array in the second.
    """
    if levels is None:
        return np.asarray(simulate(src, freqs), dtype=float)
    return np.asarray([simulate(src, freqs, level) for level in levels],
                      dtype=float)


def residual(a, b, *, align=True):
    """How far apart two signatures are in SHAPE, in dB RMS.

    'align' removes the best constant offset first, and is on because a
    measured leg never arrives at the deck's absolute level: the
    interface has its own gain and the board has its own output level,
    and neither is what is being diagnosed.  Without it a level error of
    a few dB swamps every shape difference.

    A ladder is aligned ROW BY ROW.  Each rung was measured at its own
    drive, through whatever gain the bench had at the time, so one offset
    across the whole ladder would fold those differences into the
    residual and rank on them.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if align:
        d = d - (d.mean(axis=-1, keepdims=True) if d.ndim > 1 else d.mean())
    return float(np.sqrt(np.mean(d * d)))


def candidates(src, freqs, simulate, refs=None, factors=(), *, levels=None):
    """Every single-component fault, simulated once, as [(ref, factor, db)].

    Separate from the matching because the library does not depend on the
    measurement: a bench simulates this once for a board it knows, then
    diagnoses as many builds as walk past.  The nominal circuit is in here
    as (None, 1.0) so a board that is simply correct can say so rather
    than being made to pick a scapegoat.
    """
    if refs is None:
        refs = sorted(components(src))
    out = [(None, 1.0, signature(src, freqs, simulate, levels))]
    for ref in refs:
        out.extend((ref, f, signature(perturb(src, ref, f), freqs, simulate, levels))
                   for f in factors)
    return out


def match(cands, measured, *, align=True):
    """Rank a library against one measurement: [(residual_db, ref, factor)]."""
    out = [(residual(db, measured, align=align), ref, f) for ref, f, db in cands]
    out.sort(key=lambda r: r[0])
    return out


def localise(src, freqs, measured, simulate, refs=None, factors=(), *,
             align=True, levels=None):
    """candidates() then match(), for a caller diagnosing exactly one board."""
    return match(candidates(src, freqs, simulate, refs, factors, levels=levels),
                 measured, align=align)


#
# Ranking is not the same as answering.
#
# match() always returns a closest candidate, and something is always
# closest.  On a rig the deck does not quite describe - a stray
# capacitance, a source impedance nobody wrote down - the closest
# candidate is a real part at a real factor, with a comfortable margin
# over the runner-up, and it is wrong.  Measured: 2.2 nF of undeclared
# cable capacitance on a HEALTHY board blames a resistor at ten times
# nominal, 0.06 dB clear of second place.
#
# So the residual has to be read against something.  The measurement's own
# repeatability is that something: the true candidate can only sit about
# one noise floor away from the measurement, so a best candidate sitting
# far outside it means the answer is not in the dictionary at all.
#
#
# An open and a short are not exotic faults, they are the common ones: a
# cold joint and a bridged pad.  Neither is a value being wrong by a
# factor, so a grid that only spans 0.1x to 10x cannot represent the two
# things most likely to be true of a board that does not work.  They cost
# nothing to add - a resistor going open is a huge one, a capacitor going
# open is a tiny one, and the solver takes both.
#
OPEN = 1e9
SHORT = 1e-9
FACTORS = (SHORT, 0.1, 0.22, 0.47, 2.2, 4.7, 10.0, OPEN)


def describe(factor):
    """'OPEN', 'SHORT', or 'x4.7' - what to print for a factor."""
    if factor >= OPEN / 1e3:
        return "OPEN"
    if factor <= SHORT * 1e3:
        return "SHORT"
    return f"x{factor:g}"


UNEXPLAINED = 3.0

#
# ...and a floor under it, because the method has its own error even on a
# perfect bench.  Extraction costs about 0.004 dB, and the factor grid is
# coarse, so a fault landing between two of its rungs is short by more
# than nothing.  0.02 dB is clear of both and still far under the 0.10 dB
# that undeclared stray capacitance produced.
#
FLOOR_DB = 0.02


def explain(cands, measured, *, noise_db=0.02, margin_db=0.02, align=True):
    """Rank, and then say whether the top answer is worth believing.

    'noise_db' is the measurement's own RMS repeatability, which is a
    thing to measure rather than guess: sweep twice without touching
    anything and take the RMS difference.

    verdict is one of
      nominal      the board matches the deck
      fault        one part explains it, clear of the runner-up
      ambiguous    a part fits, but not better than the next answer
      unexplained  nothing here fits; the deck does not describe the rig
    """
    ranked = match(cands, measured, align=align)
    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else (float("inf"), None, 1.0)
    margin = runner[0] - best[0]

    if best[0] > max(UNEXPLAINED * noise_db, FLOOR_DB):
        verdict = "unexplained"
    elif margin < margin_db:
        verdict = "ambiguous"
    elif best[1] is None:
        verdict = "nominal"
    else:
        verdict = "fault"

    return {"verdict": verdict, "ref": best[1], "factor": best[2],
            "residual": best[0], "margin": margin, "ranked": ranked}


def resolution(cands, *, align=True):
    """How far each fault sits from the healthy response, in dB.

    A part the output barely depends on produces a candidate that is
    nearly the nominal curve, and no bench can tell the two apart.  That
    is the circuit's property and not the bench's fault: 32 ohms of change
    in a 100 ohm source feeding a 10k load moves the output by 0.006 dB,
    and nothing will ever measure it.
    """
    nominal = next(db for ref, _f, db in cands if ref is None)
    return sorted((residual(db, nominal, align=align), ref, f)
                  for ref, f, db in cands if ref is not None)


def resolvable(cands, noise_db, *, align=True):
    """Drop the faults this bench could not see even if they were there.

    Leaving them in does not make the answer safer.  It makes a healthy
    board look ambiguous against a fault nobody could have detected, which
    reads as a warning and is only arithmetic.
    """
    floor = max(UNEXPLAINED * noise_db, FLOOR_DB)
    keep = {(ref, f) for d, ref, f in resolution(cands, align=align) if d > floor}
    return [c for c in cands if c[0] is None or (c[0], c[1]) in keep]
