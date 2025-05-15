## Demodulates subliminal back to an audible track

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

# === Low-pass filter using SOS ===
def lowpass_filter(audio, sr, cutoff=3000, order=6):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    sos = butter(order, norm_cutoff, btype='low', output='sos')
    return sosfilt(sos, audio)

# === Load modulated subliminal WAV ===
sample_rate, modulated = wavfile.read("output_subliminal.wav")

# Normalize and convert to float
if modulated.dtype == np.int16:
    modulated = modulated.astype(np.float32) / 32768.0

# Mono only
if modulated.ndim > 1:
    modulated = np.mean(modulated, axis=1)

# === Step 1: Envelope detection (absolute value) ===
envelope = np.abs(modulated)

# === Step 2: Low-pass filter the envelope ===
demodulated = lowpass_filter(envelope, sample_rate, cutoff=3000)

# === Step 3: Subtract 1 to reverse modulation offset ===
demodulated -= 1.0

# === Step 4: Normalize and convert to int16 ===
demodulated /= np.max(np.abs(demodulated))
output = np.int16(demodulated * 32767)

# === Save result ===
wavfile.write("demodulated_output.wav", sample_rate, output)

print("Demodulated voice saved as 'demodulated_output.wav'")