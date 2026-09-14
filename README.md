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

## Limits

- **One fault at a time.** Two parts wrong at once is a much larger search
  and a much weaker claim. A board with two faults usually announces
  itself anyway.
- **Linear devices in `mna`.** R, C, L and an independent voltage source.
  Anything else goes to a real simulator.
- **Not validated against a physical board.** Every number here comes from
  an injected fault. The open question is whether a real measurement, with
  its noise floor and converter distortion, clears the budget above.
