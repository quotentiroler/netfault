"""Reporting shared by the test scripts.

Each one runs as its own process, so the failure list is per-run.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILED = []


def deck(name):
    """A netlist from this directory, by filename."""
    return (HERE / name).read_text()


def record(name, ok, detail=""):
    print(f"  {name:<52} {'ok' if ok else 'FAIL'}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def summary(who, note=""):
    """Say what failed, and hand back the exit code."""
    if FAILED:
        print(f"{who}: {len(FAILED)} FAILED")
        for name in FAILED:
            print(f"    {name}")
        return 1
    print(f"{who}: ok{note}")
    return 0
