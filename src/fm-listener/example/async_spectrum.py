#!/usr/bin/env python3
import asyncio
import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import welch
import dsp_helper
import threading

try:
    from rtlsdr import RtlSdr
except Exception:
    RtlSdr = None

# Defaults
CENTER_FREQ = 88.5e6     # Hz
SAMPLE_RATE = 1_024_000  # Hz
BANDWIDTH   = 200e3      # Hz
NFFT = dsp_helper.nextPower2(64*1024) # FFT size
UPDATE_HZ = 50           # UI refresh cap (Hz)

async def spectrum_async(fc: float=CENTER_FREQ, fs: int=SAMPLE_RATE, bandwidth: float=BANDWIDTH,
                         nfft: int=NFFT, window: str="hamming", update_hz: float=UPDATE_HZ) -> None:
    if RtlSdr is None:
        raise RuntimeError("pyrtlsdr not available. Install 'pyrtlsdr' and attach an RTL-SDR device.")

    sdr = RtlSdr()
    sdr.sample_rate = fs
    sdr.center_freq = fc
    sdr.bandwidth = bandwidth
    sdr.gain = 0

    # Initialize plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    # Baseband frequency axis from Welch/FFT conventions (two-sided, shifted)
    freqs_bb = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))  # Hz, from -fs/2..+fs/2
    freqs_plot = (freqs_bb + fc) / 1e6  # MHz absolute
    line, = ax.plot(freqs_plot, np.full(nfft, -120.0, dtype=np.float32), lw=1.0)
    ax.set_xlim(freqs_plot[0], freqs_plot[-1])
    ax.set_ylim(-130, 10)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('PSD (dB/Hz)')
    ax.set_title(f'Welch PSD @ {fc/1e6:.3f} MHz, fs={fs/1e6:.3f} MS/s, NFFT={nfft}')
    ax.grid(True)

    # Accumulate samples to maintain at least nfft coverage
    buf = np.zeros(0, dtype=np.complex64)

    # UI update pacing
    min_dt = 1.0 / max(1.0, float(update_hz))
    last_draw = 0.0

    # Graceful shutdown when plot window is closed
    stop_flag = threading.Event()

    def _on_close(_event):
        stop_flag.set()

    cid = fig.canvas.mpl_connect('close_event', _on_close)

    try:
        async for chunk in sdr.stream():
            # Exit quickly if window closed
            if stop_flag.is_set() or not plt.fignum_exists(fig.number):
                try:
                    await sdr.stop()
                except Exception:
                    pass
                break
            x = np.asarray(chunk, dtype=np.complex64)
            if x.size == 0:
                await asyncio.sleep(0)
                continue
            # Append and keep only last ~2*NFFT for flexibility
            buf = np.concatenate([buf, x])
            if buf.size > (2 * nfft):
                buf = buf[-2*nfft:]

            # Compute PSD via Welch once we have at least one NFFT segment
            if buf.size < nfft:
                await asyncio.sleep(0)
                continue

            f, Pxx = welch(buf, fs=fs, window=window, nperseg=nfft, noverlap=nfft // 2,
                           nfft=nfft, return_onesided=False, scaling="density", average="mean")

            # Shift to center DC and convert to dB/Hz
            psd_db = dsp_helper.get_psd_db(Pxx)

            # Update plot
            now = time.perf_counter()
            if (now - last_draw) >= min_dt:
                line.set_ydata(psd_db)
                fig.canvas.draw_idle()
                plt.pause(0.01)
                last_draw = now

            # If the window was closed during the pause, stop streaming
            if stop_flag.is_set() or not plt.fignum_exists(fig.number):
                try:
                    await sdr.stop()
                except Exception:
                    pass
                break

            await asyncio.sleep(0)
    finally:
        try:
            fig.canvas.mpl_disconnect(cid)
        except Exception:
            pass
        try:
            await sdr.stop()
        except Exception:
            pass
        sdr.close()
        try:
            plt.ioff()
        except Exception:
            pass

def main():
    print("Starting spectrum analysis...")
    try:
        asyncio.run(spectrum_async())
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
