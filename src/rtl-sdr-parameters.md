# RTL‑SDR (R820T/2 + RTL2832U) – Configurable Parameters & Ranges

> This cheatsheet targets the common **RTL‑SDR Blog V3** (R820T2 tuner + RTL2832U). Ranges are driver/tuner‑dependent.

## Core Specs (hardware limits)

- **Tuning range (RF tuner mode):** ~**24 MHz – 1.766 GHz** (R820T/2).  
- **HF (direct sampling, V3):** **~0.5 MHz – ≈24 MHz** usable (you can tune up to 28.8 MHz but expect mirror/alias images).  
- **ADC resolution:** **8‑bit** (RTL2832U).  
- **TCXO accuracy (V3):** ~**1 ppm** class; typical drift ~0.5–1 ppm; initial offset ~0–2 ppm.  
- **Built‑in bias‑tee (V3):** **~4.5 V**, ≈**180 mA** max (software‑switchable).

## Sampling & Bandwidth

- **Supported sample‑rate ranges (librtlsdr):**  
  - **225,001 – 300,000 S/s**  
  - **900,001 – 3,200,000 S/s**  
- **Stable upper rate:** plan for **2.40–2.56 MS/s** (3.2 MS/s often drops samples).  
- **Usable complex baseband bandwidth:** ≈ **sample rate** (e.g., 2.4 MS/s ≈ 2.4 MHz span).

## Available Parameters
Full descriptions available at pyrtlsdr's [readthedocs](https://pyrtlsdr.readthedocs.io/en/latest/rtlsdr.html).

| Parameter | Typical Values / Notes |
|---|---|
| **Center frequency** | 24 MHz–1.766 GHz (tuner) · 0.5–≈24 MHz (direct sampling, Q‑branch) |
| **Sample rate** | 225 k–300 k or 900 k–3.2 M S/s (use ≤2.56 M for reliability) |
| **RF gain** | 0.0–~49.6 dB in discrete steps; or **auto** |
| **PPM correction** | Integer PPM |

---

**Applies to:** RTL‑SDR Blog **V3** (R820T2 + RTL2832U).
