"""From a KiCad schematic to a named part, without touching hardware.

The other tests start from a netlist somebody wrote by hand. This one
starts where a real board starts: rc2.kicad_sch is a schematic KiCad
itself can open, built from the symbol definitions KiCad ships, and the
deck under test is whatever kicad-cli exports from it. So the thing being
checked is the whole path, including the half that belongs to KiCad.

What is still simulated is the bench. The measurement is a solve of a
deck with one value edited, not a sweep of a board on a desk, so this
says nothing about whether a real rig matches its netlist. It does say
that everything between the schematic and the verdict is sound.

The edit is deliberately a different code path from perturb(): the
measurement comes from rewriting the deck's text, the dictionary comes
from netfault's own perturbation, and they have to agree.

Skipped when kicad-cli is not installed.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netfault
from netfault import mna
from netfault.values import parse_value

HERE = Path(__file__).resolve().parent
SCHEMATIC = HERE / "rc2.kicad_sch"
FREQS = list(np.geomspace(100.0, 6400.0, 7))
FACTORS = (0.1, 0.47, 2.2, 10.0)
NODE = "/out"
NOISE = 0.005
FAILED = []

WINDOWS_VERSIONS = ("9.0", "8.0", "7.0")


def record(name, ok, detail=""):
    print(f"  {name:<52} {'ok' if ok else 'FAIL'}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def kicad_cli():
    """The CLI, wherever this platform keeps it."""
    found = shutil.which("kicad-cli") or shutil.which("kicad-cli.exe")
    if found:
        return found
    for base in (r"C:\Program Files\KiCad", r"C:\Program Files (x86)\KiCad"):
        for version in WINDOWS_VERSIONS:
            path = Path(base) / version / "bin" / "kicad-cli.exe"
            if path.is_file():
                return str(path)
    for path in ("/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
                 "/usr/bin/kicad-cli", "/usr/local/bin/kicad-cli"):
        if Path(path).is_file():
            return path
    return None


def run(cli, *args):
    return subprocess.run([cli, *args], capture_output=True, text=True,
                          timeout=120, check=False)


def export(cli):
    """kicad-cli sch export netlist --format spice, as kicad-mcp runs it."""
    out = Path(tempfile.mkdtemp()) / "rc2.cir"
    done = run(cli, "sch", "export", "netlist", "--format", "spice", "-o", str(out),
               str(SCHEMATIC))
    if done.returncode != 0 or not out.is_file():
        msg = f"kicad-cli could not export the fixture: {(done.stderr or done.stdout).strip()}"
        raise RuntimeError(msg)
    return out.read_text(encoding="utf-8")


def erc(cli):
    """KiCad's own opinion of the fixture."""
    out = Path(tempfile.mkdtemp()) / "erc.rpt"
    done = run(cli, "sch", "erc", "--exit-code-violations", "-o", str(out), str(SCHEMATIC))
    return done.returncode == 0


def value_of(src, ref):
    return parse_value(re.search(rf"(?m)^{ref}\s+\S+\s+\S+\s+(\S+)", src).group(1))


def plant(src, ref, factor):
    return re.sub(rf"(?m)^({ref}\s+\S+\s+\S+\s+)\S+",
                  lambda m: m.group(1) + repr(value_of(src, ref) * factor), src)


def main():
    cli = kicad_cli()
    if cli is None:
        print("test-kicad: SKIPPED - no kicad-cli on this machine")
        return 0

    print(f"test-kicad: {cli}")
    record("KiCad's own ERC passes the fixture", erc(cli))
    src = export(cli)

    devices = {ref for _k, ref, _a, _b, _v in mna.devices(src)}
    record("kicad-cli exports every part", devices == {"R1", "R2", "C1", "C2", "V1"},
           " ".join(sorted(devices)))
    record("the ground net is SPICE node 0",
           any(b == "0" or a == "0" for _k, _r, a, b, _v in mna.devices(src)))
    record("the output node survives the export", NODE in src)

    refs = sorted(netfault.components(src))
    record("only passives are blameable", refs == ["C1", "C2", "R1", "R2"], " ".join(refs))

    sim = mna.runner(NODE)
    cands = netfault.candidates(src, FREQS, sim, refs, netfault.FACTORS)
    rng = np.random.default_rng(7)

    def measure(deck):
        return sim(deck, FREQS) + rng.normal(0.0, NOISE, len(FREQS))

    print()
    print("  a fault is planted by editing the exported deck, and has to come back")
    missed = []
    for ref in refs:
        for factor in FACTORS:
            v = netfault.explain(cands, measure(plant(src, ref, factor)),
                                 noise_db=netfault.FLOOR_DB)
            if v["ref"] != ref or abs(v["factor"] - factor) > 1e-9:
                missed.append("{} {} -> {} {}".format(
                    ref, netfault.describe(factor), v["ref"],
                    "-" if v["ref"] is None else netfault.describe(v["factor"])))
    record("every planted fault names its own part and factor", not missed,
           f"{len(refs) * len(FACTORS) - len(missed)} of {len(refs) * len(FACTORS)}")
    for m in missed:
        print(f"      {m}")

    v = netfault.explain(cands, measure(src), noise_db=netfault.FLOOR_DB)
    record("a healthy board is called nominal", v["verdict"] == "nominal", v["verdict"])

    stray = src.replace(".end", "Cstray /a 0 2.2n\n.end")
    v = netfault.explain(cands, measure(stray), noise_db=netfault.FLOOR_DB)
    record("a fault outside the dictionary is refused", v["verdict"] == "unexplained",
           "{} at {:.4f} dB".format(v["verdict"], v["residual"]))

    print()
    if FAILED:
        print(f"test-kicad: {len(FAILED)} FAILED")
        for f in FAILED:
            print(f"    {f}")
        return 1
    print("test-kicad: ok - schematic to named part, no hardware")
    return 0


if __name__ == "__main__":
    sys.exit(main())
