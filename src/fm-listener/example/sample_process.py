#!/usr/bin/env python3
import numpy as np
import wave
import dsp_helper
from scipy.signal import resample_poly

SAMPLE_RATE = 1_024_000  # Hz front-end capture rate
AUDIO_RATE = 48_000      # Hz output audio rate
DECIM = 10               # first stage decimation (1.024e6 -> 102.4 kHz)
DEEMPH_TAU = 75e-6       # 75 us (North America); use 50 us for Europe
IN_FILE = 'samples.dat'
OUT_FILE = 'processed.wav'

MIN_INT16 = np.iinfo(np.int16).min
MAX_INT16 = np.iinfo(np.int16).max


def process(iq_file: str = IN_FILE, fs_in: int = SAMPLE_RATE, decim: int = DECIM,
            audio_rate: int = AUDIO_RATE, out_wav: str = OUT_FILE, play: bool = True) -> None:
    # Read full file
    iq = dsp_helper.read_iq_file(iq_file)

    # Demodulate FM signal
    demod_iq = dsp_helper.fm_demod(iq)

    # Decimate
    x = resample_poly(x=demod_iq, up=1, down=decim, window="hamming")
    fs_inter = fs_in // decim

    # Resample to audio
    up, down = dsp_helper.calc_up_down(fs_inter, audio_rate)
    audio = resample_poly(x=x, up=up, down=down, window="hamming")

    # De-emphasis (broadcast FM pre-emphasis reversal)
    audio = dsp_helper.deemphasis(audio=audio, fs=audio_rate, tau=DEEMPH_TAU)

    # Normalize
    normalized_audio = dsp_helper.normalize(audio)

    # Write WAV file
    with wave.open(out_wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(audio_rate))
        a = normalized_audio.astype(np.int16) # convert to int16
        a16 = np.clip(a=a, a_min=MIN_INT16, a_max=MAX_INT16).astype(np.int16)
        wf.writeframes(a16.tobytes())

    # Optional Playback
    if play:
        try:
            import sounddevice as sd
            sd.play(data=audio, samplerate=int(audio_rate))
            sd.wait()
        except Exception:
            pass

def main():
    process()

if __name__ == '__main__':
    main()
