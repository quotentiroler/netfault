#
# SPICE value spellings, in one place because three files want them.
#
import re

SI = {"f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6,
      "m": 1e-3, "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12}

_NUM = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
_VALUE = re.compile(rf"^({_NUM})\s*([a-zA-Z]*)$")


def parse_value(text):
    """'4.7k' -> 4700.0, '100p' -> 1e-10, '1meg' -> 1e6.

    SPICE reads the suffix greedily and ignores what trails it, so '100pF'
    is '100p' and '1kohm' is '1k'.  'meg' is tested before 'm' because 'm'
    is milli here, and taking it first is a billion-fold error that still
    looks like a number.
    """
    m = _VALUE.match(text.strip())
    if not m:
        msg = f"not a plain value: {text!r}"
        raise ValueError(msg)
    num, suffix = float(m.group(1)), m.group(2).lower()
    if not suffix:
        return num
    if suffix.startswith("meg"):
        return num * SI["meg"]
    return num * SI.get(suffix[0], 1.0)


def format_value(v):
    """A float back into something a deck will read the same way."""
    return f"{v:.6g}"
