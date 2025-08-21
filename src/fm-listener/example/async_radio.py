#!/usr/bin/env python3
import asyncio
import dsp_helper
import numpy as np
import sounddevice as sd
from typing import Tuple
from scipy.signal import resample_poly, lfilter

# RTL-SDR
try:
    from rtlsdr import RtlSdr
except Exception:
    RtlSdr = None

# Defaults
CENTER_FREQ = 88.5e6    # Hz
SAMPLE_RATE = 1_024_000 # Hz
BANDWIDTH   = 200e3     # Hz
GAIN        = 0
AUDIO_RATE  = 48_000    # Hz

# Upsample/downsample declaration
DECIM = 10 # decimation factor
UP, DOWN = dsp_helper.calc_up_down(SAMPLE_RATE//DECIM, AUDIO_RATE)


def fm_demod(iq: np.ndarray, prev: complex) -> Tuple[np.ndarray, complex]:
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

class Deemph:
    """
    De-emphasis filter for FM demodulated audio.
    """
    def __init__(self, fs_audio: int, tau: float = 75e-6) -> None:
        self.fs = int(fs_audio)
        self.tau = float(tau)  # time constant (seconds)
        # Pole mapping
        self.alpha = float(np.exp(-1.0 / (self.fs * self.tau)))  # pole factor
        # Difference equation: y[n] = (1-alpha) * x[n] + alpha * y[n-1]
        self.b = np.array([1.0 - self.alpha], dtype=np.float64)
        self.a = np.array([1.0, -self.alpha], dtype=np.float64)
        self.zi = np.array([0.0], dtype=np.float64) # lfilter state
    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Apply de-emphasis filter to the input signal.
        """
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)
        x = np.asarray(x, dtype=np.float64, order='C') # ensure contiguous, row-major storage
        y, zf = lfilter(self.b, self.a, x, zi=self.zi)
        self.zi = zf  # persist state across chunks
        return y.astype(np.float32, copy=False)

class AudioRing:
    """
    Circular buffer for audio data.
    """
    def __init__(self, rate: int, seconds: float = 1.0):
        buf_len = int(rate * seconds)
        self.buf = np.zeros(buf_len, dtype=np.float32)
        self.w = 0 # write index
        self.r = 0 # read index
        self.count = 0 # number of valid samples in buffer
        self.lock = asyncio.Lock()
    async def write(self, x: np.ndarray) -> None:
        """
        Write audio data to the ring buffer.

        Overwrite behavior: if there isn't enough free space, the oldest samples
        are dropped to make room for the new incoming samples.
        """
        if x is None or x.size == 0:
            return
        x = np.asarray(x, dtype=np.float32)
        # Number of samples in the incoming chunk
        chunk_len = int(x.size)
        async with self.lock:
            # Total buffer capacity
            cap = self.buf.size

            # If the incoming chunk is larger than the buffer, keep only the most recent
            if chunk_len > cap:
                x = x[-cap:]
                chunk_len = cap

            # Free space currently available in the buffer
            free_space = cap - self.count

            # If not enough space, drop the oldest samples to make room
            if chunk_len > free_space:
                drop_count = chunk_len - free_space
                self.r = (self.r + drop_count) % cap
                self.count -= drop_count

            # Number of samples we can write before wrapping
            first_chunk = min(cap - self.w, chunk_len)
            self.buf[self.w:self.w+first_chunk] = x[:first_chunk]

            # Remaining samples to wrap to the beginning
            second_chunk = chunk_len - first_chunk
            if second_chunk > 0:
                self.buf[0:second_chunk] = x[first_chunk:]

            # Advance write index and update count
            self.w = (self.w + chunk_len) % cap
            self.count += chunk_len

    def read(self, n: int) -> np.ndarray:
        """
        Read audio data from the ring buffer.

        Underrun behavior: if fewer samples are available than requested,
        the remainder of the output is zero-padded.
        """
        # Number of samples requested by the caller
        requested = int(n)
        # Output buffer is prefilled with zeros for any underrun
        out = np.zeros(requested, dtype=np.float32)
        # Total buffer capacity
        cap = self.buf.size
        
        # No lock in audio callback (avoid blocking); tolerate slight races
        # How many samples are actually available to read now
        to_copy = min(requested, self.count)
        # First contiguous segment before wrap
        first_chunk = min(cap - self.r, to_copy)
        out[:first_chunk] = self.buf[self.r:self.r+first_chunk]
        # Remainder that wraps to index 0
        second_chunk = to_copy - first_chunk
        if second_chunk > 0:
            out[first_chunk:first_chunk+second_chunk] = self.buf[0:second_chunk]
        # Advance read index and decrement count by what we consumed
        self.r = (self.r + to_copy) % cap
        self.count -= to_copy
        return out

async def run_radio() -> None:
    if RtlSdr is None:
        raise RuntimeError("pyrtlsdr not available. Install 'pyrtlsdr' and attach an RTL-SDR device.\nLinux tip: sudo apt-get install rtl-sdr python3-pyrtlsdr")

    # Initialize audio processing components
    deemph = Deemph(AUDIO_RATE)
    last = 0+0j
    ring = AudioRing(rate=AUDIO_RATE, seconds=1.0)

    def audio_cb(outdata: np.ndarray, frames: int, time: None, status: None) -> None:
        """
        Audio callback for playback.

        Parameters
        ----------
        outdata : ndarray
            Output audio data.
        frames : int
            Number of frames to process.
        time : None
            Mandatory for OutputStream callback, not used.
        status : None
            Mandatory for OutputStream callback, not used.
        """
        y = ring.read(frames)
        outdata[:, 0] = y

    # Initialize audio output stream
    stream = sd.OutputStream(samplerate=AUDIO_RATE, channels=1, dtype='float32', blocksize=1024, callback=audio_cb)

    # Initialize RTL-SDR
    sdr = RtlSdr()
    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = CENTER_FREQ
    sdr.bandwidth = BANDWIDTH
    sdr.gain = GAIN

    async def process_chunk(chunk: np.ndarray):
        """
        Process a chunk of audio data.
        """
        nonlocal last
        x = chunk.astype(np.complex64)

        # FM demod, with continuity
        dem, last = fm_demod(x, last)

        # Decimate
        x = resample_poly(x=dem, up=1, down=DECIM, window="hamming")

        # Resample to audio
        y = resample_poly(x=x, up=UP, down=DOWN, window="hamming")

        # De-emphasis & soft clip
        y = deemph.process(y)
        y = np.tanh(y * 2.5).astype(np.float32)
        await ring.write(y)

    stream.start()
    try:
        # Async process sample chunks
        async for chunk in sdr.stream():
            await process_chunk(np.asarray(chunk))
    finally:
        try:
            await sdr.stop()
        except Exception:
            pass
        sdr.close()
        try:
            stream.stop(); stream.close()
        except Exception:
            pass

def main():
    try:
        asyncio.run(run_radio())
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
