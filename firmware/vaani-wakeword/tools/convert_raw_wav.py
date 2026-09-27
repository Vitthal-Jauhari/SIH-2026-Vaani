import numpy as np
import wave

input_file = "test-servers/audio_chunk.raw"
output_file = "test-servers/audio.wav"

SAMPLE_RATE = 16000

# Read raw 32-bit samples
samples = np.fromfile(input_file, dtype=np.int32)

print("Samples:", len(samples))
print("Duration:", len(samples) / SAMPLE_RATE, "seconds")

# INMP441 data is typically 24-bit data inside a 32-bit I2S word.
# Shift it down to get usable audio.
samples = samples >> 8

# Convert 24-bit-ish values to signed 16-bit PCM
samples = np.clip(samples, -32768, 32767).astype(np.int16)

# Write WAV
with wave.open(output_file, "wb") as wav:
    wav.setnchannels(1)       # Mono
    wav.setsampwidth(2)       # 16-bit
    wav.setframerate(SAMPLE_RATE)
    wav.writeframes(samples.tobytes())

print("Created:", output_file)