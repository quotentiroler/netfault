# netfault

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22744875.svg)](https://doi.org/10.5281/zenodo.22744875)

Measure a circuit, and find out **which component is wrong**.

Comparing a build against its schematic and reporting "you disagree above
2 kHz" is the easy half. What somebody holding a soldering iron needs is a
reference designator.

```
$ python tests/run.py
localise  (inject a fault, then find it)
  names the part and the factor, every time        ok  14/14
  a correct board is called correct                ok  named None
```

## Running it

With [uv](https://docs.astral.sh/uv/), nothing to install and no venv to make:

```bash
uv run examples/bench.py examples/breadboard.cir --simulate --fault C2:4.7
```

The script carries its own dependencies inline (PEP 723), so that command
works on a fresh clone. For the library:

```bash
uv pip install -e .          # or: pip install -e .
uv run tests/run.py
```

`numpy` is the only runtime dependency. `sounddevice` is needed just by
the bench example, because the library and both test suites never touch a
device - which is what the `play_record` seam is for.

## How it works

Rather than inverting the measurement, every single-component fault that
could explain it is simulated, and the closest one wins.

That is brute force, and it is completely robust: no derivative, no
assumption that the response is linear in the part values, and no way to
be fooled by a topology it was not told about, because every candidate
*is* the topology.

Two things make it practical:

- **The candidates are simulated once.** They do not depend on the
  measurement, so a bench builds the library for a board it knows and then
  diagnoses as many builds as walk past.
- **Ranking is on shape, not level.** A measured leg never arrives at the
  netlist's absolute level, because the interface has its own gain and the
  board has its own output level. The best constant offset is removed
  before comparing. Without that, a +6 dB offset is enough to make the
  wrong part win outright.

## No simulator required

For a linear network the answer is a solve rather than a program, so
`netfault.mna` does modified nodal analysis in numpy: exact, instant, and
no subprocess. The whole test suite runs in under a second on python and
numpy alone.

`simulate` is injectable, so a circuit with diodes or op-amps in it can be
handed to ngspice through the same interface:

```python
cands = netfault.candidates(src, freqs, simulate, refs, factors)
ranked = netfault.match(cands, measured)
print(ranked[0])          # (residual_db, 'C2', 4.7)
```

## What it needs from a measurement

The ranking separates candidates by fractions of a dB, so the measurement
has to be good. Measured against a 3-pole RC ladder, top-1 holds at 100%
through 0.20 dB RMS of noise. On a denser circuit it is tighter: against a
23-component pedal netlist, 99% at 0.05 dB RMS and 79% at 0.10, because
many perturbations of a busy circuit look alike.

**About 0.05 dB RMS is the budget to aim at**, and that is a real
requirement on the bench rather than a formality.

## Measuring the board

You need a sweep of levels. `netfault.measure` does stepped sine, which is
slower than a chirp and has nothing to get subtly wrong: one tone, one
correlation, everything that is not that tone discarded.

`play_record` is the seam. It takes a stimulus and returns what came back,
and nothing above it cares whether that is a sound card, a pedal over USB
audio, or a filter standing in for one in a test.

```python
import sounddevice as sd
from netfault import measure

def play_record(x, fs):
    return sd.playrec(x, samplerate=fs, channels=1, blocking=True)[:, 0]

ref  = measure.calibrate(play_record, freqs)      # output looped to input
meas = measure.response(play_record, freqs)       # board in the loop
ranked = netfault.match(cands, meas - ref)
print(ranked[0])                                  # (0.03, 'C2', 4.7)
```

Calibrate first with the output looped straight back, and subtract. That
step is what decides whether the budget is met: the extraction itself is
good to about 0.004 dB, so everything left over is the interface.

## Your first bench

`examples/breadboard.cir` is deliberately the cheapest circuit that has
anything to say: a two-pole RC low-pass, four jellybean parts, corners in
the audio band. Two 10k resistors, two 10n capacitors, a breadboard, two
jack leads and any interface that records while it plays.

`Rsrc` and `Rin` in that deck are the interface's own output and input
impedance rather than parts you fit, which is why they are excluded below.

**1. Before building anything**, check the install against the netlist:

```bash
uv run examples/bench.py examples/breadboard.cir --simulate --fault C2:4.7
```

It should name `C2` at `4.7x`. If it does not, stop here; no wiring will
fix it.

**2. Build the divider**, then run it for real. One invocation asks for
the loopback first and the board second:

```bash
uv run examples/bench.py examples/breadboard.cir --exclude Rsrc,Rin
```

**3. Read the repeatability figure it prints before anything else.** That
is the noise floor every later threshold is measured against. If it is far
above 0.02 dB RMS, the answer is in the cabling or the interface, and
nothing downstream will be trustworthy until it comes down.

**4. A healthy board has to come back `nominal`.** This is the step worth
being stubborn about. If a board you know is correct reads `unexplained`,
the deck does not describe the rig, and almost always the missing piece is
the source and load impedance. Fix the netlist, not the board. If it reads
as a fault with a comfortable margin, that is the failure mode documented
under [When the answer is not a part](#when-the-answer-is-not-a-part), and
it is the reason that section exists.

**5. Only then plant one.** Swap a 10n for a 47n and it should name `C2`
at about `4.7x`. A fault found before step 4 passed is not evidence of
anything.

## Circuits that change with drive

A linear network is the same network at every level, so one sweep says
everything about it. A circuit with a clipper in it is not, and the part
that decides where it folds does nothing at all until the drive reaches
it. Ranking on one sweep cannot tell a wrong one from a right one, and it
does not fail loudly when it cannot: it ranks on the noise instead.

So a signature can be a **ladder**, one response per drive level:

```python
LEVELS = (-36.0, -24.0, -12.0)
cands = netfault.candidates(src, freqs, simulate, refs, factors, levels=LEVELS)
meas  = netfault.signature(src, freqs, simulate, levels=LEVELS)
```

`simulate` is then called as `simulate(src, freqs, level)`, and a candidate
is a `(level, freq)` array rather than a curve. Measured on a test network
whose clipping stage only engages above a knee:

| | one sweep at -40 dB | ladder at -40/-20/-6 |
|---|---|---|
| the clipper's own part | within 0.0036 dB of every rival, under the floor | named |
| verdict | `ambiguous` | `fault` |
| all parts | — | 12/12 |

Each rung is aligned **on its own**. It was measured at its own drive
through whatever gain the bench had at the time, so one offset across the
whole ladder would fold those differences into the residual and rank on
them.

## When the answer is not a part

`match()` always returns a closest candidate, because something always is
closest. On a rig the deck does not quite describe, that candidate is a
real part at a real factor with a comfortable margin, and it is wrong.
Measured: **2.2 nF of undeclared cable capacitance on a healthy board
blames a resistor at ten times nominal, 0.06 dB clear of second place.**

So read the residual against the bench's own repeatability, which is a
thing to measure rather than guess:

```python
noise = measure.repeatability(play_record, freqs)   # sweep twice, untouched
v = netfault.explain(cands, meas, noise_db=noise)
v["verdict"]    # nominal | fault | ambiguous | unexplained
```

`unexplained` means nothing in the dictionary accounts for the
measurement, and the deck needs fixing before any part is believed.

Two more things stop a healthy board being blamed:

- `resolvable()` drops faults this bench could not have seen anyway. A
  part the output barely depends on produces a candidate that is nearly
  the healthy curve, and offering it as a rival is arithmetic rather than
  information.
- Parts that are not on the board cannot be mis-fitted. The interface's
  own source and load impedance belong in the deck, and belong out of the
  candidate set: `--exclude Rsrc,Rin`.

## Relation to prior work

The method here is a **fault dictionary** - the simulation-before-test
approach to analog fault diagnosis: simulate a catalogue of faults, store
their signatures, and match a measurement against the catalogue. It is not
new. The standard survey is Bandler and Salama, "Fault Diagnosis of Analog
Circuits", Proc. IEEE 73 (1985), 1279-1325, and there is a modern taxonomy
in [AEU 2016](https://www.sciencedirect.com/science/article/abs/pii/S1434841116308445).

What is here that is not in a paper:

- a runnable MIT-licensed implementation with tests, `pip install -e .`
- a numpy MNA solver, so a linear dictionary builds with no simulator
- a shape-aligned residual, because a measured leg arrives at its own level
- an `unexplained` verdict, so an incomplete model yields no answer rather
  than a confident wrong one
- opens and shorts in the dictionary beside parametric drift, since a cold
  joint and a bridged pad are the common faults and neither is a value
  being off by a factor
- a signature that can be a ladder, so a circuit whose behaviour depends
  on drive can be diagnosed at all

If you are benchmarking a new diagnosis method, this is meant to serve as
the classical baseline rather than yet another private reimplementation of
one. If it is wrong or unfair as a baseline, that is worth an issue.

Citable: [10.5281/zenodo.22744875](https://doi.org/10.5281/zenodo.22744875) resolves to the
latest version, and GitHub's "Cite this repository" has the formatted
entry.

## Acknowledgement

The measurement approach here was developed while working with the
`Validation` suite in [torvalds/GuitarPedal](https://github.com/torvalds/GuitarPedal),
which solves the same problem for one specific piece of hardware.
`netfault` shares no code with it - that project is GPL-2.0 and this one
is MIT - but the stepped-sine approach, the loopback calibration and the
settling behaviour were all informed by reading `loop.py` and `audio.py`
there first.

## Limits

- **One fault at a time.** Two parts wrong at once is a much larger search
  and a much weaker claim. A board with two faults usually announces
  itself anyway.
- **Linear devices in `mna`.** R, C, L and an independent voltage source.
  A nonlinear circuit goes to a real simulator through the same
  `simulate` seam, and wants a ladder rather than a sweep.
- **A dictionary is slow for nonlinear circuits.** One ngspice transient
  is around 1.8 s, so 23 parts at eight factors over three rungs and eight
  points is about two hours. It is built once per board and matched in
  milliseconds after that, which is why `candidates()` and `match()` are
  separate.
- **Not validated against a physical board.** Every number here comes from
  an injected fault. The open question is whether a real measurement, with
  its noise floor and converter distortion, clears the budget above.
  [Your first bench](#your-first-bench) is four parts and an afternoon, and
  answering it does not need anything more than that.
