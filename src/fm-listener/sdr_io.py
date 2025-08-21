#! /usr/bin/env python3
from __future__ import annotations
import numpy as np

try:
    from rtlsdr import RtlSdr
except Exception:
    RtlSdr = None

DEFAULT_FS = 1_024_000
DEFAULT_BLOCK = DEFAULT_FS // 10  # samples per read

class SDRDevice:
    def __init__(self, sample_rate: int = DEFAULT_FS):
        if RtlSdr is None:
            raise RuntimeError("pyrtlsdr not available. Install 'pyrtlsdr' and attach an RTL-SDR device.")
        self.sdr = RtlSdr()
        self.sdr.sample_rate = int(sample_rate)
        self.sdr.gain = 0
        self.sdr.bandwidth = 200_000  # Default FM bandwidth
        self.blocksize = DEFAULT_BLOCK

    def set_center_freq(self, freq_hz: float):
        self.sdr.center_freq = float(freq_hz)

    def get_center_freq(self) -> float:
        return float(self.sdr.center_freq)

    def read_block(self) -> np.ndarray:
        """
        Read one block of IQ (complex64).

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            Block of IQ samples.
        """
        block = self.sdr.read_samples(self.blocksize)
        return np.asarray(block, dtype=np.complex64)

    async def stream(self):
        """Async generator that yields blocks of IQ samples (complex64)."""
        async for block in self.sdr.stream():
            yield np.asarray(block, dtype=np.complex64)

    async def stop_stream(self):
        """Stop the async stream if running."""
        try:
            await self.sdr.stop()
        except Exception:
            pass

    def close(self):
        try:
            self.sdr.close()
        except Exception:
            pass
