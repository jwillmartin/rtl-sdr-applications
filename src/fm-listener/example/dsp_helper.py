import math
import numpy as np
from typing import Tuple
from scipy.signal import lfilter

def read_iq_file(iq_file: str) -> np.ndarray:
    """
    Read IQ data from a file.

    Parameters
    ----------
        iq_file (str): Path to the IQ file.

    Returns
    -------
        np.ndarray: Array of complex64 IQ samples.
    """
    iq = np.fromfile(iq_file, dtype=np.complex64)
    if iq.size == 0:
        raise ValueError('Input file is empty')
    return iq

def calc_up_down(f_in: int, f_target: int) -> Tuple[int, int]:
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

def get_blocks(fs: int, duration: int, block_duration: float) -> Tuple[float, int, int, int]:
    """
    Get block parameters for processing.

    Parameters
    ----------
        fs (int): Sample rate.
        duration (int): Total duration in seconds.
        block_duration (float): Block duration in seconds.

    Returns
    -------
        Tuple[float, int, int, int]: Total samples, block samples, total number of blocks, remainder samples.
    """
    total_samples     = fs * duration
    block_samples     = int(fs * block_duration)
    total_num_blocks  = total_samples // block_samples
    remainder_samples = total_samples % block_samples

    return total_samples, block_samples, total_num_blocks, remainder_samples

def nextPower2(n: int) -> int:
    """
    Find the next power of 2 greater than or equal to n.

    Parameters
    ----------
        n (int): Input integer.

    Returns
    -------
        int: Next power of 2 greater than or equal to n.
    """
    if n <= 0:
        return 0
    else:
        log2n = math.log2(n)
        nextPow = math.ceil(log2n)
        return 2**nextPow

def get_psd_db(Pxx: np.ndarray) -> np.ndarray:
    """
    Shift to DC and convert the power spectral density (PSD) to dB/Hz from the given power spectrum.

    Parameters
    ----------
        Pxx (numpy.ndarray): Input power spectrum.

    Returns
    -------
        numpy.ndarray: PSD in dB/Hz.
    """
    Pxx_shift = np.fft.fftshift(Pxx)
    return 10.0 * np.log10(Pxx_shift + 1e-12)

def fm_demod(x: np.ndarray) -> np.ndarray:
    """
    Perform FM demodulation on a complex baseband signal.

    Parameters
    ----------
        x (np.ndarray): Input complex baseband signal.

    Returns
    -------
        np.ndarray: FM demodulated output signal.
    """
    if x.size <= 1:
        return np.zeros(0, dtype=np.float32)
    y = x[1:] * np.conj(x[:-1])
    return np.angle(y).astype(np.float32)

def get_overlap(Nblk: int, overlap: float, x: np.ndarray) -> Tuple[int, int]:
    """
    Get the number of samples to overlap between segments.

    Parameters
    ----------
        Nblk (int): Block size for segmenting the data.
        overlap (float): Overlap ratio between segments (0 to 1).
        x (numpy.ndarray): Input signal.

    Returns
    -------
        int: Number of overlapping samples.
    """
    overlap = int(Nblk * (1 - overlap))
    length = int(len(x)/overlap) - 1
    return overlap, length

def normalize(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio signal to the range [-1, 1].

    Parameters
    ----------
        audio (numpy.ndarray): Input audio signal.

    Returns
    -------
        numpy.ndarray: Normalized audio signal.
    """
    if audio.size == 0:
        return audio
    m = float(np.max(np.abs(audio)))
    if m > 0:
        audio /= m
    return audio

def deemphasis(audio: np.ndarray, fs: int, tau: float = 75e-6) -> np.ndarray:
    """
    Apply FM broadcast de-emphasis.

    Parameters
    ----------
    audio : np.ndarray
        Input demodulated (and typically audio-rate) signal.
    fs : int
        Sample rate of the audio signal (e.g., 48000).
    tau : float, optional
        Time constant in seconds. 75e-6 for NA, 50e-6 for EU.

    Returns
    -------
    np.ndarray
        De-emphasized audio (float32).
    """
    if audio.size == 0:
        return audio
    # Pole mapping (impulse invariant) for analog RC with time constant tau.
    alpha = np.exp(-1.0 / (fs * tau))  # pole factor per sample
    # Difference equation: y[n] = (1-alpha) * x[n] + alpha * y[n-1]
    b = [1.0 - alpha]
    a = [1.0, -alpha]
    y = lfilter(b, a, audio)
    y = np.asarray(y)
    return y.astype(np.float32, copy=False)
