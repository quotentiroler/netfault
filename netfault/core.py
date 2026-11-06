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
import warnings
from itertools import combinations

import numpy as np

from .values import format_si, format_value, parse_value

_DEVICE = re.compile(r"^([RCL]\w*)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)\s*$", re.MULTILINE)


class UnsolvableCandidate(UserWarning):
    """A perturbation the simulator would not converge on."""


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

    A candidate 'simulate' refuses is dropped with an UnsolvableCandidate
    warning naming it, and the rest of the dictionary is still built.  The
    nominal one is not optional and its failure is raised.
    """
    known = components(src)
    if refs is None:
        refs = sorted(known)

    # A ref that is not in the deck is the caller's mistake, not a
    # simulator that would not converge, and must not be skipped as one.
    unknown = sorted(set(refs) - set(known))
    if unknown:
        msg = f"not in the netlist: {', '.join(unknown)}"
        raise ValueError(msg)

    # Not optional: if this will not solve, simulate() is broken.
    out = [(None, 1.0, signature(src, freqs, simulate, levels))]

    # SHORT and OPEN can leave a circuit a simulator will not converge on.
    # One lost answer beats a lost build.
    for ref in refs:
        for factor in factors:
            db = _trial(src, ref, factor, freqs, simulate, levels)
            if db is not None:
                out.append((ref, factor, db))
    return out


def _trial(src, ref, value, freqs, simulate, levels):
    try:
        return signature(build(src, ref, value), freqs, simulate, levels)
    except Exception as e:  # noqa: BLE001 - simulate() is the caller's, and raises anything
        warnings.warn(f"{ref} {describe(value, ref)} will not simulate: {e}",
                      UnsolvableCandidate, stacklevel=3)
        return None


STRAY = (10e-12, 22e-12, 47e-12, 100e-12, 220e-12, 470e-12, 1e-9, 2.2e-9, 4.7e-9, 10e-9)
BRIDGE = (1.0, 10.0, 100.0, 1e3, 1e4)

_END = re.compile(r"^\.end\s*$", re.MULTILINE | re.IGNORECASE)


def added(ref):
    """Whether a candidate is a part the deck does not have: '+C(out)', '+R(a,b)'."""
    return isinstance(ref, str) and ref.startswith("+")


def attach(src, line):
    """The deck with one more device line, before its .end."""
    ends = list(_END.finditer(src))
    if not ends:
        return f"{src.rstrip()}\n{line}\n"
    at = ends[-1].start()
    return f"{src[:at]}{line}\n{src[at:]}"


def build(src, ref, value):
    """The deck a candidate describes: a part scaled by 'value', or an added part of 'value'."""
    if not added(ref):
        return perturb(src, ref, value)
    kind, nodes = ref[1], ref[3:-1].split(",")
    a, b = (nodes[0], "0") if kind == "C" else nodes
    parts = components(src)
    name = f"{kind}netfault"
    while name in parts:
        name += "_"
    return attach(src, f"{name} {a} {b} {format_value(value)}")


def additions(src, freqs, simulate, *, nodes=None, stray=STRAY, bridges=BRIDGE,
              levels=None):
    """Faults that are not a part being wrong, as [(ref, value, db)].

    '+C(node)' is 'value' farads from a node to ground: cable, breadboard,
    an interface's input.  '+R(a,b)' is 'value' ohms between two nodes no
    part joins: a solder bridge.  Add these to candidates() and an
    'unexplained' board can come back naming what the deck is missing.
    """
    parts = components(src)
    wired = {frozenset(p["nodes"]) for k, p in parts.items() if k[0].upper() == "R"}
    found = sorted(set(nodes or {n for p in parts.values() for n in p["nodes"]}) - {"0"})
    plans = [(f"+C({n})", v) for n in found for v in stray]
    plans += [(f"+R({a},{b})", v) for a, b in combinations([*found, "0"], 2)
              if frozenset((a, b)) not in wired for v in bridges]
    out = []
    for ref, value in plans:
        db = _trial(src, ref, value, freqs, simulate, levels)
        if db is not None:
            out.append((ref, value, db))
    return out


REFINE_SPAN = 3.5
REFINE_STEPS = 30


def refine(src, freqs, simulate, measured, ref, value, *, levels=None, align=True):
    """Slide one candidate's value to where it fits best: (residual_db, ref, value).

    A dictionary is a grid, and a real fault is rarely on it: 1.5 nF of
    cable sits between 1 nF and 2.2 nF and fits neither.  This searches
    value / REFINE_SPAN to value * REFINE_SPAN on a log scale (golden
    section), which reaches across the widest step of every grid here.
    """
    span, steps = REFINE_SPAN, REFINE_STEPS
    def cost(x):
        db = signature(build(src, ref, 10.0 ** x), freqs, simulate, levels)
        return residual(db, measured, align=align)

    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    lo, hi = np.log10(value / span), np.log10(value * span)
    c, d = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fc, fd = cost(c), cost(d)
    for _ in range(steps):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - ratio * (hi - lo)
            fc = cost(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + ratio * (hi - lo)
            fd = cost(d)
    best = c if fc < fd else d
    return (min(fc, fd), ref, float(10.0 ** best))


def _once(ranked):
    best = {}
    for r in sorted(ranked, key=lambda r: r[0]):
        best.setdefault(r[1], r)
    return sorted(best.values(), key=lambda r: r[0])


NEAR_NOMINAL = 1.02


def sharpen(src, freqs, simulate, measured, ranked, *, top=None, levels=None, align=True):
    """refine() the best 'top' refs of a ranking (all by default), keeping each ref once.

    Keeping each ref once makes the runner-up a different explanation
    rather than the same part one grid step over.  OPEN and SHORT are left
    as they are: they are states, not values to slide.  A board whose best
    answer is already nominal is left alone, and a part that slides back to
    within 2% of its own value is dropped, because that is nominal again.
    With ngspice every refine is about 30 simulations, so pass a 'top'.
    """
    out = _once(ranked)
    if out[0][1] is None:
        return out
    slidable = [r for r in out if r[1] is not None and (added(r[1]) or not _extreme(r[2]))]
    for _r, ref, value in slidable[:top]:
        try:
            r = refine(src, freqs, simulate, measured, ref, value, levels=levels, align=align)
        except Exception as e:  # noqa: BLE001 - simulate() is the caller's, and raises anything
            warnings.warn(f"{ref} would not refine: {e}", UnsolvableCandidate, stacklevel=2)
            continue
        if added(ref) or abs(np.log(r[2])) > np.log(NEAR_NOMINAL):
            out.append(r)
    return _once(out)


def _admittance(kind, value):
    return value if kind == "C" else 1.0 / value


def twins(src, ref, value):
    """Other answers no measurement can tell from (ref, value), as [(ref, value)].

    Parts of one kind across the same two nodes add up, so 1.5 nF more from
    'out' to ground IS C3 being 2.5x when C3 is 1 nF from 'out' to ground.
    """
    parts = components(src)
    if added(ref):
        kind, nodes = ref[1], ref[3:-1].split(",")
        pair = frozenset((nodes[0], "0") if kind == "C" else nodes)
        extra = _admittance(kind, value)
    elif ref in parts and not _extreme(value):
        kind, pair = ref[0].upper(), frozenset(parts[ref]["nodes"])
        own = parts[ref]["value"]
        extra = _admittance(kind, own * value) - _admittance(kind, own)
    else:
        return []
    out = []
    for other, p in sorted(parts.items()):
        if other == ref or other[0].upper() != kind or frozenset(p["nodes"]) != pair:
            continue
        y = _admittance(kind, p["value"]) + extra
        if y > 0:
            out.append((other, _admittance(kind, y) / p["value"]))
    named = sorted(pair - {"0"}) + sorted(pair & {"0"})
    if not added(ref) and extra > 0 and (kind == "R" or named[-1] == "0"):
        label = named[0] if kind == "C" else ",".join(named)
        out.append((f"+{kind}({label})", _admittance(kind, extra)))
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


def describe(factor, ref=None):
    """'OPEN', 'SHORT', 'x4.7', or for an added part its value: '2.2nF', '1ohm'.

    A capacitor scaled up is a short and scaled down is an open, the other
    way round from a resistor or an inductor, so pass 'ref' to get it right.
    """
    if added(ref):
        return format_si(factor) + ("F" if ref.startswith("+C") else "ohm")
    flip = isinstance(ref, str) and ref[:1].upper() == "C"
    if factor >= OPEN / 1e3:
        return "SHORT" if flip else "OPEN"
    if factor <= SHORT * 1e3:
        return "OPEN" if flip else "SHORT"
    return f"x{factor:g}"


def _extreme(factor):
    return factor >= OPEN / 1e3 or factor <= SHORT * 1e3


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
    return judge(match(cands, measured, align=align), noise_db=noise_db, margin_db=margin_db)


def judge(ranked, *, noise_db=0.02, margin_db=0.02):
    """explain()'s verdict on a ranking that is already made."""
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


def diagnose(src, freqs, simulate, measured, *, noise_db, refs=None, factors=FACTORS,
             rig=True, top=None, levels=None, align=True, margin_db=0.02):
    """The whole question for one board: dictionary, rig additions, refine, verdict.

    With 'rig' the dictionary also holds what the deck does not (stray
    capacitance, solder bridges), and the 'top' answers are slid off the
    grid to their best value before the verdict is reached.
    """
    cands = candidates(src, freqs, simulate, refs, factors, levels=levels)
    if rig:
        cands += additions(src, freqs, simulate, levels=levels)
    ranked = match(resolvable(cands, noise_db, align=align), measured, align=align)
    return conclude(src, freqs, simulate, measured, ranked, noise_db=noise_db, top=top,
                    levels=levels, align=align, margin_db=margin_db)


def conclude(src, freqs, simulate, measured, ranked, *, noise_db, top=None, levels=None,
             align=True, margin_db=0.02):
    """sharpen() a ranking, fold the winner's twins into it, and judge().

    The twins come back under 'twins' instead of as the runner-up, since
    an answer cannot be ambiguous against itself.
    """
    ranked = sharpen(src, freqs, simulate, measured, ranked, top=top, levels=levels,
                     align=align)
    _r, ref, value = ranked[0]
    same = twins(src, ref, value) if ref is not None else []
    alias = {r for r, _v in same}
    v = judge([r for r in ranked if r[1] not in alias],
              noise_db=noise_db, margin_db=margin_db)
    v["twins"] = same
    return v


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
