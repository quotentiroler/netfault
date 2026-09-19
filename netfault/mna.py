#
# Small-signal AC by modified nodal analysis.
#
# A linear network has a closed-form answer at every frequency, so asking
# a simulator for one is a subprocess, a temporary directory and a parsed
# text file to learn something a 6x6 solve already knows.  Fault-finding
# asks for hundreds of these, which is where the difference stops being
# stylistic.
#
# Linear devices only - R, C, L and an independent voltage source.  A
# circuit with a diode or a transistor in it is not this function's
# business and should be handed to ngspice through the same interface.
#
import re

import numpy as np

from .values import parse_value

_DEV = re.compile(r"^([RCLV])(\w+)\s+(\S+)\s+(\S+)\s+(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


def devices(src):
    """[(kind, ref, node+, node-, value)] for the linear devices in a deck."""
    out = []
    for m in _DEV.finditer(src):
        kind, ref = m.group(1).upper(), m.group(1) + m.group(2)
        tail = m.group(5).strip()
        if kind == "V":
            tail = re.sub(r"^(AC|DC)\s+", "", tail, flags=re.IGNORECASE)
        try:
            value = parse_value(tail.split()[0])
        except (ValueError, IndexError):
            continue
        out.append((kind, ref, m.group(3), m.group(4), value))
    return out


def solve(src, node, freqs):
    """Gain at 'node' in dB, from the deck's 1 V AC source, at each freq."""
    devs = devices(src)
    names = sorted({n for _k, _r, a, b, _v in devs for n in (a, b)} - {"0"})
    idx = {n: i for i, n in enumerate(names)}
    sources = [d for d in devs if d[0] == "V"]
    n, m = len(names), len(sources)

    def at(w):
        A = np.zeros((n + m, n + m), dtype=complex)
        b = np.zeros(n + m, dtype=complex)

        def stamp(a, c, y):
            for p, q, s in ((a, a, y), (c, c, y), (a, c, -y), (c, a, -y)):
                if p in idx and q in idx:
                    A[idx[p], idx[q]] += s

        for kind, _ref, a, c, v in devs:
            if kind == "R":
                stamp(a, c, 1.0 / v)
            elif kind == "C":
                stamp(a, c, 1j * w * v)
            elif kind == "L":
                stamp(a, c, 1.0 / (1j * w * v) if w else 1e12)
        for k, (_kind, _ref, a, c, v) in enumerate(sources):
            row = n + k
            if a in idx:
                A[row, idx[a]] += 1.0
                A[idx[a], row] += 1.0
            if c in idx:
                A[row, idx[c]] -= 1.0
                A[idx[c], row] -= 1.0
            b[row] = v
        return np.linalg.solve(A, b)[idx[node]]

    out = [at(2.0 * np.pi * f) for f in freqs]
    return 20.0 * np.log10(np.maximum(np.abs(out), 1e-30))


def runner(node):
    """A simulate(src, freqs) bound to one output node."""
    return lambda src, freqs: solve(src, node, freqs)
