#
# Measuring a board with whatever interface is already on the desk.
#
# Stepped sine rather than a swept one.  A sweep is faster and a chirp
# deconvolution is the usual answer, but the number that matters here is
# a level at a frequency to a hundredth of a dB, and the stepped version
# gets there by construction: one tone, one bin, coherent over a whole
# number of cycles, with everything that is not that tone discarded.  It
# is slower and there is nothing to get subtly wrong.
#
# 'play_record' is the seam.  It takes a stimulus and returns what came
# back, and this file does not care whether that is a sound device, a
# pedal over USB audio, or a filter standing in for one during a test.
#
import numpy as np


def tone(f, seconds, fs, amplitude=0.5):
    """A sine at f, trimmed to a whole number of cycles.

    Whole cycles so the DFT bin is exact and nothing leaks into its
    neighbours, which is what lets level_at() be a single bin rather than
    a windowed estimate.
    """
    n = max(int(round(seconds * fs / (fs / f))) , 1) * int(round(fs / f))
    t = np.arange(n) / float(fs)
    return amplitude * np.sin(2.0 * np.pi * f * t)


def level_at(x, f, fs):
    """The level of the component at f, in dBFS.

    Correlated against the exact frequency rather than read off an FFT
    bin, so it does not matter whether the capture is a whole number of
    cycles - which it will not be, once a device has trimmed or padded it.
    """
    x = np.asarray(x, dtype=float)
    t = np.arange(len(x)) / float(fs)
    w = 2.0 * np.pi * f * t
    i = 2.0 * np.mean(x * np.cos(w))
    q = 2.0 * np.mean(x * np.sin(w))
    return 20.0 * np.log10(max(np.hypot(i, q), 1e-30))


def response(play_record, freqs, fs=48000, seconds=0.25, amplitude=0.5,
             settle=0.02):
    """Level at each frequency, in dB, one tone at a time.

    'settle' drops the head of each capture, where a device is still
    starting up and an analog stage is still charging.  Measuring that is
    measuring the equipment.
    """
    out = []
    for f in freqs:
        x = tone(f, seconds, fs, amplitude)
        y = np.asarray(play_record(x, fs), dtype=float)
        skip = min(int(settle * fs), max(len(y) // 4, 0))
        out.append(level_at(y[skip:], f, fs) - level_at(x[skip:], f, fs))
    return np.asarray(out)


def calibrate(play_record, freqs, **kw):
    """The interface's own response, measured with its output looped back.

    Subtract it from a measurement of the board and what is left is the
    board.  This is the step that decides whether the budget is met: a
    plain interface is flat to a few tenths of a dB, and a few tenths is
    already more than the fault ranking can spare.
    """
    return response(play_record, freqs, **kw)
