## Demodulates subliminal back to an audible track

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

# === Filters using SOS (matching modulate.py) ===
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
    """Apply filter multiple times for sharper cutoff"""
    result = audio.copy()
    for _ in range(iterations):
        result = filter_func(result, sr, cutoff)
    return result

# === Load modulated subliminal WAV ===
sample_rate, modulated = wavfile.read("output_subliminal.wav")

# Normalize and convert to float
if modulated.dtype == np.int16:
    modulated = modulated.astype(np.float32) / 32768.0

# Mono only
if modulated.ndim > 1:
    modulated = np.mean(modulated, axis=1)

# === Reverse modulation parameters (same as in modulate.py) ===
carrier_freq = 17500  # Hz (must match the original modulation)
carrier_freq = max(14000, min(carrier_freq, 20000))  # Clamp to valid range

# Calculate cutoff frequency (same logic as modulate.py)
cutoff = min(carrier_freq / 2.0, (sample_rate / 2.0) - carrier_freq)

print(f"Demodulating with carrier frequency: {carrier_freq} Hz")
print(f"Using cutoff frequency: {cutoff} Hz")

# === Step 1: Generate the same carrier wave ===
t = np.linspace(0, len(modulated) / sample_rate, len(modulated), endpoint=False)
carrier = np.sin(2 * np.pi * carrier_freq * t)

# === Step 2: Demodulate by multiplying with carrier again ===
# This shifts the modulated signal back to baseband
demodulated = modulated * carrier

# === Step 3: Low-pass filter to extract the original signal ===
# Remove high-frequency components and keep only the baseband signal
demodulated_lp = iterative_filter(demodulated, sample_rate, cutoff, lowpass_filter, iterations=10)

# === Step 4: Remove DC component and scale ===
# The demodulation process introduces a DC offset and scaling
demodulated_lp = demodulated_lp - np.mean(demodulated_lp)

# === Step 5: Scale back (reverse the "mult 2" from modulation) ===
demodulated_lp = demodulated_lp / 2.0

# === Step 6: Apply high-pass filter at 80 Hz (reverse the original HP filter) ===
# This removes any remaining DC and low-frequency artifacts
final_audio = highpass_filter(demodulated_lp, sample_rate, 80, order=8)

# === Step 7: Normalize and convert to int16 ===
if np.max(np.abs(final_audio)) > 0:
    final_audio /= np.max(np.abs(final_audio))

output = np.int16(final_audio * 32767)

# === Save result ===
wavfile.write("demodulated_output.wav", sample_rate, output)

print("Demodulated voice saved as 'demodulated_output.wav'")
print("Original audio signal has been recovered from subliminal track")