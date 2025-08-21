#! /usr/bin/env python3
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
from scipy.signal import resample_poly, lfilter


@dataclass
class FMState:
    fs_in: int = 1_024_000
    fs_audio: int = 48_000
    last_sample: complex = 0+0j
    deemp_y: float = 0.0
    tau: float = 75.0e-6  # NA de-emphasis, 50.0e-6 for EU
    # Persistent de-emphasis filter state (zi for lfilter)
    deemp_zi: np.ndarray = field(default_factory=lambda: np.array([0.0], dtype=np.float64))

def fm_demod_quad(iq: np.ndarray, prev: complex) -> tuple[np.ndarray, complex]:
    """
    Quadrature discriminator demodulation with continuity.

    Parameters
    ----------
    iq : np.ndarray
        Input IQ samples.
    prev : complex
        Previous sample for continuity.

    Returns
    -------
    tuple[np.ndarray, complex]
        Demodulated samples and last sample.
    """
    if iq.size == 0:
        return np.zeros(0, dtype=np.float32), prev
    y = np.empty_like(iq, dtype=np.complex64)
    y[0] = iq[0] * np.conj(prev)
    y[1:] = iq[1:] * np.conj(iq[:-1])
    dem = np.angle(y).astype(np.float32)
    return dem, complex(iq[-1])

def deemphasis(x: np.ndarray, state: FMState) -> np.ndarray:
    """
    Simple single-pole de-emphasis filter.

    Parameters
    ----------
    x : np.ndarray
        Input audio samples.
    state : FMState
        FM state containing filter parameters.

    Returns
    -------
    np.ndarray
        De-emphasized audio samples.
    """
    # Pole mapping
    alpha = float(np.exp(-1.0 / (state.fs_audio * state.tau)))  # pole factor
    # Difference equation: y[n] = (1-alpha) * x[n] + alpha * y[n-1]
    b = np.array([1.0 - alpha], dtype=np.float64)
    a = np.array([1.0, -alpha], dtype=np.float64)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    x = np.asarray(x, dtype=np.float64, order='C')  # ensure contiguous, row-major storage
    # Use persistent state across blocks
    y, zf = lfilter(b, a, x, zi=state.deemp_zi)
    state.deemp_zi = zf  # persist state across chunks
    return y.astype(np.float32, copy=False)

def calc_up_down(f_in: int, f_target: int) -> tuple[int, int]:
    """
    Compute the up and down sampling factors for a given input and target frequency.

    Parameters
    ----------
        f_in (int): Input frequency.
        f_target (int): Target frequency.

    Returns
    -------
        Tuple[int, int]: Up and down sampling factors.
    """
    g = math.gcd(f_in, f_target)
    return f_target // g, f_in // g

def resample_to_audio(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """
    Resample audio to the target output sample rate.

    Parameters
    ----------
    x : np.ndarray
        Input audio signal.
    fs_in : int
        Input sample rate.
    fs_out : int
        Output sample rate.

    Returns
    -------
    np.ndarray
        Resampled audio signal.
    """
    if fs_in == fs_out:
        return x.astype(np.float32)

    # Decimate
    inter_x = resample_poly(x=x, up=1, down=10, window="hamming")
    fs_inter = fs_in // 10
    # Resample to audio
    upsample, downsample = calc_up_down(fs_inter, fs_out)
    y = resample_poly(inter_x, upsample, downsample, window="hamming")
    return y

def process_fm_block(iq: np.ndarray, state: FMState) -> np.ndarray:
    """
    Process a single block of FM IQ samples.

    Parameters
    ----------
    iq : np.ndarray
        Input IQ samples.
    state : FMState
        FM state containing processing parameters.

    Returns
    -------
    np.ndarray
        Processed audio samples.
    """
    demod, state.last_sample = fm_demod_quad(iq, state.last_sample)
    audio = resample_to_audio(demod, state.fs_in, state.fs_audio)
    audio = deemphasis(audio, state)
    # Soft clip
    audio = np.tanh(audio * 2.5).astype(np.float32)
    return audio
