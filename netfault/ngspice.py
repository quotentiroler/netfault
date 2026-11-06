import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .measure import level_at
from .values import format_value, parse_value

ENV = "NETFAULT_NGSPICE"
TIMEOUT = 600.0

_LINES = re.MULTILINE | re.IGNORECASE
_SOURCE = re.compile(r"^(V\w*)\s+(\S+)\s+(\S+)\s*(.*?)\s*$", _LINES)
_CONTROL = re.compile(r"^\.control\b.*?^\.endc\b[^\n]*\n?", _LINES | re.DOTALL)
_END = re.compile(r"^\.end\s*$", _LINES)


class SpiceError(RuntimeError):
    """ngspice ran and did not produce the answer it was asked for."""


def executable():
    """The ngspice binary: $NETFAULT_NGSPICE, else the console build, else plain ngspice."""
    named = os.environ.get(ENV)
    if named:
        return named if Path(named).is_file() else shutil.which(named)
    return shutil.which("ngspice_con") or shutil.which("ngspice")


def available():
    return executable() is not None


def _body(src):
    return _END.split(_CONTROL.sub("", src))[0].rstrip("\n")


def run(src, commands, outputs, *, timeout=TIMEOUT):
    """Run 'commands' as a .control block on the deck; return each written file as an array.

    Every command list is expected to write the files named in 'outputs' with wrdata.
    A file that is missing afterwards means its analysis failed, and is an error.
    """
    exe = executable()
    if exe is None:
        msg = f"ngspice not found; install it or set {ENV}"
        raise SpiceError(msg)
    control = "\n".join([".control", "set wr_singlescale", *commands, "quit", ".endc"])
    deck = f"{_body(src)}\n{control}\n.end\n"
    with tempfile.TemporaryDirectory(prefix="netfault-") as tmp:
        cwd = Path(tmp)
        (cwd / "deck.cir").write_text(deck)
        try:
            subprocess.run(  # noqa: S603 - the binary is located by executable(), the deck is ours
                [exe, "-b", "-o", "deck.log", "deck.cir"], cwd=cwd, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as e:
            msg = f"ngspice did not finish in {timeout:g} s"
            raise SpiceError(msg) from e
        missing = [name for name in outputs if not (cwd / name).is_file()]
        if missing:
            log = cwd / "deck.log"
            tail = log.read_text(errors="replace")[-1500:] if log.is_file() else ""
            msg = f"ngspice wrote no {', '.join(missing)}:\n{tail}"
            raise SpiceError(msg)
        return [np.atleast_2d(np.loadtxt(cwd / name)) for name in outputs]


def ac(src, node, freqs, *, per_decade=200):
    """Small-signal gain at 'node' in dB at each freq, from the deck's own AC source."""
    f = np.asarray(freqs, dtype=float)
    lo, hi = f.min() / 1.05, f.max() * 1.05
    (data,) = run(src, [f"ac dec {per_decade} {lo:.9g} {hi:.9g}",
                        f"wrdata ac.dat v({node})"], ["ac.dat"])
    db = 20.0 * np.log10(np.maximum(np.hypot(data[:, 1], data[:, 2]), 1e-30))
    return np.interp(np.log10(f), np.log10(data[:, 0]), db)


def drive(src, level, source=None):
    """The deck with its input source replaced by a sine of 10**(level/20) V peak.

    The sine's frequency is the parameter nf_freq, so one deck serves every frequency.
    """
    found = [m for m in _SOURCE.finditer(src)
             if (m.group(1).upper() == source.upper() if source
                 else re.search(r"\bAC\b", m.group(4), re.IGNORECASE))]
    if len(found) != 1:
        what = source or "an AC source"
        msg = f"wanted exactly one {what} to drive, found {len(found)}"
        raise ValueError(msg)
    m = found[0]
    dc = re.search(r"\bDC\s+(\S+)", m.group(4), re.IGNORECASE)
    bare = re.match(r"^([-+.\d]\S*)", m.group(4))
    offset = parse_value((dc or bare).group(1)) if (dc or bare) else 0.0
    amp = format_value(10.0 ** (level / 20.0))
    line = (f".param nf_freq=1k\n{m.group(1)} {m.group(2)} {m.group(3)} "
            f"DC {offset:.9g} SIN({offset:.9g} {amp} {{nf_freq}})")
    return src[:m.start()] + line + src[m.end():]


def tran(src, node, freqs, level, *, source=None, cycles=8, per_cycle=128, settle=0.02):
    """Fundamental gain at 'node' in dB at each freq, driven at 'level' dBV peak.

    Each tone runs for 'settle' seconds (at least four cycles) before 'cycles' whole
    cycles are kept, resampled uniformly, and correlated against the tone.
    """
    deck = drive(src, level, source)
    commands, outputs, plans = [], [], []
    for i, f in enumerate(freqs):
        period = 1.0 / f
        step = period / per_cycle
        start = max(settle, 4.0 * period)
        stop = start + cycles * period
        name = f"t{i}.dat"
        commands += [f"alterparam nf_freq={f:.9g}", "reset",
                     f"tran {step:.9g} {stop:.9g} {start:.9g} {step:.9g}",
                     f"wrdata {name} v({node})", "destroy all"]
        outputs.append(name)
        plans.append((f, step, start))
    out = []
    for data, (f, step, start) in zip(run(deck, commands, outputs), plans):
        t = start + np.arange(cycles * per_cycle) * step
        y = np.interp(t, data[:, 0], data[:, 1])
        out.append(level_at(y, f, 1.0 / step) - level)
    return np.asarray(out)


def runner(node, *, per_decade=200):
    """A simulate(src, freqs) bound to one node, by ngspice AC analysis."""
    return lambda src, freqs: ac(src, node, freqs, per_decade=per_decade)


def tran_runner(node, *, level=-20.0, **kw):
    """A simulate(src, freqs, level=None) bound to one node, by ngspice transient.

    This is the one for circuits that change with drive: hand it to candidates()
    and signature() with levels=.  Without levels it drives at 'level'.
    """
    def simulate(src, freqs, at=None):
        return tran(src, node, freqs, level if at is None else at, **kw)
    return simulate
