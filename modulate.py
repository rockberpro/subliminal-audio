## Modulates audio into a subliminal (inaudible) track

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

# === Low-pass filter using SOS ===
def lowpass_filter(audio, sr, cutoff=3000, order=6):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    sos = butter(order, norm_cutoff, btype='low', output='sos')
    return sosfilt(sos, audio)

# === Load WAV ===
sample_rate, audio = wavfile.read("input.wav")

# Convert to float32 and normalize
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
elif audio.dtype == np.float32:
    audio = np.copy(audio)

# Convert to mono if stereo
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# === Apply low-pass filter ===
filtered_audio = lowpass_filter(audio, sample_rate, cutoff=3000)

# === Generate high-frequency carrier (e.g., 17 kHz) ===
carrier_freq = 17500  # Hz
t = np.linspace(0, len(filtered_audio) / sample_rate, len(filtered_audio), endpoint=False)
carrier = np.sin(2 * np.pi * carrier_freq * t)

# === AM Modulation: (1 + filtered_audio) * carrier ===
modulated = (1.0 + filtered_audio) * carrier

# Normalize to avoid clipping
modulated /= np.max(np.abs(modulated))

# Convert to int16 for saving
modulated_int16 = np.int16(modulated * 32767)

# === Save output WAV ===
wavfile.write("output_subliminal.wav", sample_rate, modulated_int16)

print("Subliminal audio generated and saved as 'output_subliminal.wav'")