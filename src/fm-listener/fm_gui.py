#!/usr/bin/env python3
"""
FM Receiver using pyrtlsdr, split into 3 modules:
- sdr_io.py : device access
- demod.py  : FM DSP helpers
- fm_gui.py : GUI app and asyncio worker with buffered audio
"""
from __future__ import annotations
import asyncio
import threading
import wave
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from typing import Any, Optional
import sounddevice as sd
from sdr_io import SDRDevice, DEFAULT_FS
from demod import FMState, process_fm_block

APP_TITLE = "RTL-SDR FM Receiver"
FM_START = 88.1
FM_END = 107.9
FM_STEP = 0.2  # MHz

MIN_INT16 = np.iinfo(np.int16).min
MAX_INT16 = np.iinfo(np.int16).max

def fm_station_list(start=FM_START, end=FM_END, step=FM_STEP):
    vals = []
    x = start
    # Ensure rounding to 0.1 to avoid float artifacts
    while x <= end + 1e-9:
        # Round to odd .1 multiples typical in US (88.5, 88.7, ...)
        vals.append(round(x, 1))
        x += step
    return vals

STATIONS = fm_station_list()


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

class AsyncWorker:
    def __init__(self, logger):
        self.log = logger
        self.fs_in = DEFAULT_FS
        self.fs_audio = 48_000
        self.fm_state = FMState(fs_in=self.fs_in, fs_audio=self.fs_audio)

        # runtime state
        self.dev: Optional[SDRDevice] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.loop_thread: Optional[threading.Thread] = None
        self.capture_task: Optional[asyncio.Task] = None
        self.proc_task: Optional[asyncio.Task] = None
        self.iq_task: Optional[asyncio.Task] = None
        self.queue: Optional[asyncio.Queue] = None

        # outputs
        self.play_audio = None
        self.sd_stream = None
        self.ring: Optional[AudioRing] = None
        self.record_wav = False
        self.wav_writer: Optional[wave.Wave_write] = None
        self.record_iq = False
        self.iq_fh: Optional[Any] = None

        # misc
        self.running = False
        self._start_stop_lock = threading.Lock()
        self._stopping = False  # graceful shutdown flag for capture loop

    # ---------------------------- public API ----------------------------
    def start(self, freq_mhz: float, outdir: str, play_audio: bool, record_wav: bool, record_iq: bool):
        with self._start_stop_lock:
            if self.running:
                self.log("Already running.")
                return
            self.running = True
            self._stopping = False

        # Outputs setup (paths & files)
        Path(outdir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = f"fm_{int(freq_mhz*10)/10:.1f}MHz_{ts}".replace('.', 'p')

        self.play_audio = bool(play_audio)
        self.record_wav = bool(record_wav)
        self.record_iq = bool(record_iq)

        if self.record_wav:
            wav_path = Path(outdir) / f"{base}.wav"
            self.wav_writer = wave.open(str(wav_path), 'wb')
            self.wav_writer.setnchannels(1)
            self.wav_writer.setsampwidth(2)
            self.wav_writer.setframerate(self.fs_audio)
            self.log(f"Recording WAV: {wav_path}")

        if self.record_iq:
            iq_path = Path(outdir) / f"{base}.c64"
            self.iq_fh = open(iq_path, 'wb')
            self.log(f"Recording IQ: {iq_path}")

        # Audio ring and stream
        self.ring = AudioRing(rate=self.fs_audio, seconds=1.0)
        if self.play_audio:
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
                y = self.ring.read(frames) if self.ring else np.zeros(frames, dtype=np.float32)
                outdata[:, 0] = y

            self.sd_stream = sd.OutputStream(
                samplerate=self.fs_audio,
                channels=1,
                dtype='float32',
                blocksize=1024,
                callback=audio_cb,
            )
            self.sd_stream.start()

        # Reset DSP state fresh
        self.fm_state = FMState(fs_in=self.fs_in, fs_audio=self.fs_audio)

        # Start asyncio loop in a background thread and kick off tasks
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.loop_thread.start()
        # Initialize async side (device + tasks)
        fut = asyncio.run_coroutine_threadsafe(self._async_start(freq_mhz), self.loop)
        try:
            fut.result(timeout=5.0)
        except Exception as e:
            self.log(f"Failed to start: {e}")
            self.stop()
            raise
        self.log(f"Started at {freq_mhz:.1f} MHz (fs={self.fs_in/1e6:.4f} MS/s).")

    def stop(self):
        with self._start_stop_lock:
            if not self.running:
                return
            self.running = False

        # Stop async tasks and device
        if self.loop:
            try:
                fut = asyncio.run_coroutine_threadsafe(self._async_stop(), self.loop)
                fut.result(timeout=5.0)
            except Exception:
                pass
            try:
                self.loop.call_soon_threadsafe(self.loop.stop)
            except Exception:
                pass
        if self.loop_thread:
            self.loop_thread.join(timeout=2.0)
        self.loop = None
        self.loop_thread = None

        # Close audio and files on main thread
        try:
            if self.sd_stream:
                self.sd_stream.stop(); self.sd_stream.close()
        except Exception:
            pass
        self.sd_stream = None

        try:
            if self.wav_writer:
                self.wav_writer.close()
        except Exception:
            pass
        self.wav_writer = None

        try:
            if self.iq_fh:
                self.iq_fh.close()
        except Exception:
            pass
        self.iq_fh = None

        self.log("Stopped.")

    def retune(self, freq_mhz: float):
        if not self.loop or not self.dev:
            return
        async def _ret():
            try:
                dev = self.dev
                assert dev is not None
                dev.set_center_freq(freq_mhz * 1e6)
                # Reset discriminator continuity to avoid clicks
                self.fm_state.last_sample = 0+0j
            except Exception as e:
                self.log(f"Tune error: {e}")
        try:
            asyncio.run_coroutine_threadsafe(_ret(), self.loop)
            self.log(f"Tuned {freq_mhz:.1f} MHz")
        except Exception as e:
            self.log(f"Tune failed: {e}")

    # ------------------------- asyncio internals -------------------------
    def _run_loop(self):
        assert self.loop is not None
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            # Cleanup any pending tasks
            try:
                pending = asyncio.all_tasks(loop=self.loop)
                for t in pending:
                    t.cancel()
                if pending:
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            self.loop.close()

    async def _async_start(self, freq_mhz: float):
        # Open device in loop thread
        try:
            self.dev = SDRDevice(sample_rate=self.fs_in)
            self.dev.set_center_freq(freq_mhz * 1e6)
        except Exception as e:
            self.log(f"SDR open failed: {e}")
            raise

        # Work queue and tasks
        self.queue = asyncio.Queue(maxsize=8)
        self.capture_task = asyncio.create_task(self._capture_loop())
        self.proc_task = asyncio.create_task(self._proc_loop())

    async def _async_stop(self):
        # Signal graceful shutdown to capture loop and wait for it to finish
        self._stopping = True
        # Request stream to stop if using async stream API
        try:
            if self.dev:
                await self.dev.stop_stream()
        except Exception:
            pass
        if self.capture_task and not self.capture_task.done():
            try:
                await self.capture_task
            except Exception:
                pass
        self.capture_task = None

        try:
            if self.dev:
                self.dev.close()
        except Exception:
            pass
        self.dev = None

        # Cancel processing task after capture has stopped
        if self.proc_task and not self.proc_task.done():
            try:
                self.proc_task.cancel()
                await self.proc_task
            except Exception:
                pass
        self.proc_task = None

        # Release queue
        self.queue = None

    async def _capture_loop(self):
        assert self.queue is not None
        try:
            dev = self.dev
            if dev is None:
                return
            async for iq in dev.stream():
                if self._stopping:
                    break
                if self.record_iq and self.iq_fh is not None:
                    data = iq.tobytes()
                    # File I/O off main loop
                    await asyncio.to_thread(self.iq_fh.write, data)
                try:
                    await self.queue.put(iq)
                except asyncio.CancelledError:
                    raise
        except asyncio.CancelledError:
            return
        except Exception as e:
            if not self._stopping:
                self.log(f"Capture error: {e}")

    async def _proc_loop(self):
        assert self.queue is not None
        try:
            while True:
                if self.queue is None:
                    break
                iq = await self.queue.get()
                try:
                    audio = process_fm_block(iq, self.fm_state)
                except Exception as e:
                    self.log(f"DSP error: {e}")
                    continue
                # Playback
                if self.play_audio and self.ring is not None:
                    await self.ring.write(audio)
                # WAV output
                if self.record_wav and self.wav_writer is not None:
                    a16 = np.clip(audio * MAX_INT16, MIN_INT16, MAX_INT16).astype(np.int16)
                    await asyncio.to_thread(self.wav_writer.writeframes, a16.tobytes())
        except asyncio.CancelledError:
            return
        except Exception as e:
            self.log(f"Process error: {e}")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.minsize(640, 420)
        self.resizable(True, True)
        self.worker = AsyncWorker(self.log)

        body = ttk.Frame(self, padding=12)
        body.pack(fill='both', expand=True)
        # Make grid resizable
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=1)

        r = 0
        ttk.Label(body, text="Frequency (MHz):").grid(row=r, column=0, sticky='e', padx=6, pady=6)
        self.freq = tk.DoubleVar(value=88.5)
        self.freq_entry = ttk.Entry(body, textvariable=self.freq, width=10)
        self.freq_entry.grid(row=r, column=1, sticky='ew')

        r += 1
        ttk.Label(body, text="Station step (US grid):").grid(row=r, column=0, sticky='e', padx=6, pady=6)
        self.prev_btn = ttk.Button(body, text="Prev ◀", width=10, command=self.prev_station)
        self.next_btn = ttk.Button(body, text="Next ▶", width=10, command=self.next_station)
        self.prev_btn.grid(row=r, column=1, sticky='w')
        self.next_btn.grid(row=r, column=2, sticky='w')

        r += 1
        ttk.Label(body, text="Output folder:").grid(row=r, column=0, sticky='e', padx=6, pady=6)
        self.outdir = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(body, textvariable=self.outdir, width=36).grid(row=r, column=1, columnspan=2, sticky='ew')
        ttk.Button(body, text="Browse…", command=self.choose_dir).grid(row=r, column=3, sticky='w', padx=6)

        r += 1
        self.play_var = tk.BooleanVar(value=True)
        self.wav_var = tk.BooleanVar(value=False)
        self.iq_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(body, text="Play audio", variable=self.play_var, state='normal').grid(row=r, column=0, sticky='w', padx=6, pady=6)
        ttk.Checkbutton(body, text="Record WAV", variable=self.wav_var).grid(row=r, column=1, sticky='w', padx=6, pady=6)
        ttk.Checkbutton(body, text="Record IQ (.c64)", variable=self.iq_var).grid(row=r, column=2, sticky='w', padx=6, pady=6)

        r += 1
        btns = ttk.Frame(body)
        btns.grid(row=r, column=0, columnspan=4, pady=12)
        self.start_btn = ttk.Button(btns, text="Start", width=12, command=self.on_start)
        self.stop_btn  = ttk.Button(btns, text="Stop", width=12, command=self.on_stop, state='disabled')
        self.start_btn.pack(side='left', padx=6)
        self.stop_btn.pack(side='left', padx=6)

        r += 1
        ttk.Label(body, text="Log:").grid(row=r, column=0, sticky='nw')
        self.log_box = tk.Text(body, width=70, height=10, state='disabled')
        body.rowconfigure(r, weight=1)
        self.log_box.grid(row=r, column=1, columnspan=3, sticky='nsew')

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def bump(self, delta_mhz: float):
        self.freq.set(round(self.freq.get() + delta_mhz, 1))
        self.worker.retune(self.freq.get())

    def prev_station(self):
        f = self.freq.get()
        try:
            idx = STATIONS.index(round(f, 1))
        except ValueError:
            idx = 0
        idx = max(0, idx - 1)
        self.freq.set(STATIONS[idx])
        self.worker.retune(self.freq.get())

    def next_station(self):
        f = self.freq.get()
        try:
            idx = STATIONS.index(round(f, 1))
        except ValueError:
            idx = 0
        idx = min(len(STATIONS) - 1, idx + 1)
        self.freq.set(STATIONS[idx])
        self.worker.retune(self.freq.get())

    def choose_dir(self):
        d = filedialog.askdirectory(initialdir=self.outdir.get(), title="Choose output folder")
        if d:
            self.outdir.set(d)

    def on_start(self):
        try:
            self.worker.start(self.freq.get(), self.outdir.get(), self.play_var.get(), self.wav_var.get(), self.iq_var.get())
            self.start_btn.configure(state='disabled')
            self.stop_btn.configure(state='normal')
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_stop(self):
        try:
            self.worker.stop()
        finally:
            self.start_btn.configure(state='normal')
            self.stop_btn.configure(state='disabled')

    def on_close(self):
        try:
            self.worker.stop()
        except Exception:
            pass
        self.destroy()

    def log(self, msg: str):
        # Ensure UI updates happen on the Tk thread
        def _append():
            self.log_box.configure(state='normal')
            self.log_box.insert('end', f"{msg}\n")
            self.log_box.see('end')
            self.log_box.configure(state='disabled')
        try:
            self.after(0, _append)
        except Exception:
            # Fallback if app is closing
            pass



def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
