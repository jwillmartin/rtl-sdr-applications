#!/usr/bin/env python3
import numpy as np
import dsp_helper

try:
    from rtlsdr import RtlSdr
except ImportError:
    RtlSdr = None

if RtlSdr is None:
    raise RuntimeError("pyrtlsdr not available. Install 'pyrtlsdr' and attach an RTL-SDR device.")
sdr = RtlSdr()

# FM capture parameters
FM_MIN_FREQ = 87.5e6  # FM band min frequency (Hz)
FM_MAX_FREQ = 108e6  # FM band max frequency (Hz)
FM_STEP_FREQ = 100e3  # FM step frequency (Hz)
FM_BANDWIDTH = 200e3  # FM bandwidth (Hz)

# Set to lowest allowable sample rate by RTL-SDR v3
# Still greater than target 2*FM_BANDWIDTH, per Nyquist Rate
SAMPLE_RATE = 1_024_000 # Hz

def main():
    # SDR parameters
    sdr.sample_rate     = SAMPLE_RATE   # Hz
    sdr.center_freq     = 88.5e6        # Hz
    sdr.bandwidth       = FM_BANDWIDTH  # Hz
    sdr.gain            = 0             # dB

    # Capture settings
    duration          = 10           # total duration in seconds
    block_duration    = 1            # seconds per read
    fs                = int(sdr.sample_rate)
    total_samples, block_samples, total_num_blocks, remainder_samples = dsp_helper.get_blocks(fs, duration, block_duration)

    print(f"Reading {block_samples} number of samples.")
    output_file = 'samples.dat'

    with open(output_file, 'wb') as f:
        # Capture full blocks
        for i in range(total_num_blocks):
            chunk = sdr.read_samples(block_samples)
            chunk = np.asarray(chunk, dtype=np.complex64)
            f.write(chunk.tobytes())
            print(f"Written block {i+1}/{total_num_blocks}")
        # Capture any leftover samples
        if remainder_samples:
            chunk = sdr.read_samples(remainder_samples)
            chunk = np.asarray(chunk, dtype=np.complex64)
            f.write(chunk.tobytes())
            print("Written final partial block")

    sdr.close()
    print(f"Capture complete: {duration}s ({total_samples} samples) > {output_file}")

if __name__ == "__main__":
    main()
