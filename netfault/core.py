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

_DEVICE = re.compile(r"^([RCL]\w*)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)\s*$", re.M)


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
        raise ValueError("%s matched %d devices, wanted 1" % (ref, len(seen)))
    return out


def residual(a, b, align=True):
    """How far apart two dB curves are in SHAPE, in dB RMS.

    'align' removes the best constant offset first, and is on because a
    measured leg never arrives at the deck's absolute level: the interface
    has its own gain and the board has its own output level, and neither
    is what is being diagnosed.  Without it a level error of a few dB
    swamps every shape difference and the ranking answers a question
    nobody asked.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if align:
        d = d - d.mean()
    return float(np.sqrt(np.mean(d * d)))


def candidates(src, freqs, simulate, refs=None, factors=()):
    """Every single-component fault, simulated once, as [(ref, factor, db)].

    Separate from the matching because the library does not depend on the
    measurement: a bench simulates this once for a board it knows, then
    diagnoses as many builds as walk past.  The nominal circuit is in here
    as (None, 1.0) so a board that is simply correct can say so rather
    than being made to pick a scapegoat.
    """
    if refs is None:
        refs = sorted(components(src))
    out = [(None, 1.0, simulate(src, freqs))]
    for ref in refs:
        for f in factors:
            out.append((ref, f, simulate(perturb(src, ref, f), freqs)))
    return out


def match(cands, measured, align=True):
    """Rank a library against one measurement: [(residual_db, ref, factor)]."""
    out = [(residual(db, measured, align), ref, f) for ref, f, db in cands]
    out.sort(key=lambda r: r[0])
    return out


def localise(src, freqs, measured, simulate, refs=None, factors=(),
             align=True):
    """candidates() then match(), for a caller diagnosing exactly one board."""
    return match(candidates(src, freqs, simulate, refs, factors), measured,
                 align)


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
UNEXPLAINED = 3.0

#
# ...and a floor under it, because the method has its own error even on a
# perfect bench.  Extraction costs about 0.004 dB, and the factor grid is
# coarse, so a fault landing between two of its rungs is short by more
# than nothing.  0.02 dB is clear of both and still far under the 0.10 dB
# that undeclared stray capacitance produced.
#
FLOOR_DB = 0.02


def explain(cands, measured, noise_db=0.02, margin_db=0.02, align=True):
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
    ranked = match(cands, measured, align)
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


def resolution(cands, align=True):
    """How far each fault sits from the healthy response, in dB.

    A part the output barely depends on produces a candidate that is
    nearly the nominal curve, and no bench can tell the two apart.  That
    is the circuit's property and not the bench's fault: 32 ohms of change
    in a 100 ohm source feeding a 10k load moves the output by 0.006 dB,
    and nothing will ever measure it.
    """
    nominal = next(db for ref, _f, db in cands if ref is None)
    return sorted((residual(db, nominal, align), ref, f)
                  for ref, f, db in cands if ref is not None)


def resolvable(cands, noise_db, align=True):
    """Drop the faults this bench could not see even if they were there.

    Leaving them in does not make the answer safer.  It makes a healthy
    board look ambiguous against a fault nobody could have detected, which
    reads as a warning and is only arithmetic.
    """
    floor = max(UNEXPLAINED * noise_db, FLOOR_DB)
    keep = {(ref, f) for d, ref, f in resolution(cands, align) if d > floor}
    return [c for c in cands if c[0] is None or (c[0], c[1]) in keep]
