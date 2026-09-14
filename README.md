# netfault

Measure a linear circuit, and find out **which component is wrong**.

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

If you are benchmarking a new diagnosis method, this is meant to serve as
the classical baseline rather than yet another private reimplementation of
one. If it is wrong or unfair as a baseline, that is worth an issue.

## Limits

- **One fault at a time.** Two parts wrong at once is a much larger search
  and a much weaker claim. A board with two faults usually announces
  itself anyway.
- **Linear devices in `mna`.** R, C, L and an independent voltage source.
  Anything else goes to a real simulator.
- **Not validated against a physical board.** Every number here comes from
  an injected fault. The open question is whether a real measurement, with
  its noise floor and converter distortion, clears the budget above.
