## Modulates audio into a subliminal (inaudible) track

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

# === Filters using SOS ===
def lowpass_filter(audio, sr, cutoff, order=8):
    """Low-pass filter with specified cutoff frequency"""
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    if norm_cutoff >= 1.0:
        return audio  # No filtering needed
    sos = butter(order, norm_cutoff, btype='low', output='sos')
    return sosfilt(sos, audio)

def highpass_filter(audio, sr, cutoff, order=8):
    """High-pass filter with specified cutoff frequency"""
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    if norm_cutoff <= 0.0:
        return audio  # No filtering needed
    sos = butter(order, norm_cutoff, btype='high', output='sos')
    return sosfilt(sos, audio)

def iterative_filter(audio, sr, cutoff, filter_func, iterations=10):
    """Apply filter multiple times for sharper cutoff (like Nyquist's cut function)"""
    result = audio.copy()
    for _ in range(iterations):
        result = filter_func(result, sr, cutoff)
    return result

# === Load WAV ===
sample_rate, audio = wavfile.read("input.wav")

# Validate sample rate (minimum 44100 Hz like in Nyquist code)
if sample_rate < 44100:
    raise ValueError("The track sample frequency must be minimum 44100Hz.")

# Convert to float32 and normalize
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
elif audio.dtype == np.float32:
    audio = np.copy(audio)

# Convert to mono if stereo
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# === Subliminal processing (following Nyquist logic) ===
carrier_freq = 17500  # Hz (default, can be adjusted between 14000-20000)
carrier_freq = max(14000, min(carrier_freq, 20000))  # Clamp to valid range

# Calculate cutoff frequency (following Nyquist logic)
# Must not exceed carrier/2 or (sample_rate/2 - carrier) to avoid aliasing
cutoff = min(carrier_freq / 2.0, (sample_rate / 2.0) - carrier_freq)

print(f"Carrier frequency: {carrier_freq} Hz")
print(f"Cutoff frequency: {cutoff} Hz")

# 1. Apply high-pass filter at 80 Hz (remove low frequencies)
audio_hp = highpass_filter(audio, sample_rate, 80, order=8)

# 2. Apply iterative low-pass filter with calculated cutoff
audio_filtered = iterative_filter(audio_hp, sample_rate, cutoff, lowpass_filter, iterations=10)

# 3. Generate carrier wave
t = np.linspace(0, len(audio_filtered) / sample_rate, len(audio_filtered), endpoint=False)
carrier = np.sin(2 * np.pi * carrier_freq * t)

# 4. Modulate: multiply by 2 and then by carrier (following Nyquist: mult 2 ... hzosc)
modulated = 2.0 * audio_filtered * carrier

# 5. Apply high-pass filter at carrier frequency to remove lower frequencies
modulated_hp = iterative_filter(modulated, sample_rate, carrier_freq, highpass_filter, iterations=10)

# Normalize to avoid clipping
if np.max(np.abs(modulated_hp)) > 0:
    modulated_hp /= np.max(np.abs(modulated_hp))

# Convert to int16 for saving
modulated_int16 = np.int16(modulated_hp * 32767)

# === Save output WAV ===
wavfile.write("output_subliminal.wav", sample_rate, modulated_int16)

print("Subliminal audio generated and saved as 'output_subliminal.wav'")
print(f"Modulated signal contains frequencies above {carrier_freq} Hz")