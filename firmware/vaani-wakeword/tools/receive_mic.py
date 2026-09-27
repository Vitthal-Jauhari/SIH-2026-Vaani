import serial
import wave
import time
import sys

PORT = "COM3"
BAUD = 921600

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
NUM_SAMPLES = 80000

EXPECTED_BYTES = NUM_SAMPLES * SAMPLE_WIDTH

MAX_WAIT_SECONDS = 15  # give up and report instead of hanging forever

print(f"Opening {PORT} at {BAUD} baud...")

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=2
)

# Give ESP32 time to reset after serial connection
time.sleep(1)

print("Waiting for AUDIO_START... (raw bytes below as they arrive)")
print("-" * 60)

buffer = b""
start_time = time.time()

while b"DATA\n" not in buffer:
    chunk = ser.read(256)

    if chunk:
        # Echo everything so you can SEE what's actually coming through,
        # instead of staring at a silent terminal.
        sys.stdout.write(chunk.decode("utf-8", errors="replace"))
        sys.stdout.flush()
        buffer += chunk
    else:
        if time.time() - start_time > MAX_WAIT_SECONDS:
            print("\n" + "-" * 60)
            print(f"No 'DATA' marker after {MAX_WAIT_SECONDS}s.")
            print(f"Total bytes received so far: {len(buffer)}")
            if len(buffer) == 0:
                print("Nothing arrived at all.")
                print("-> Check: ESP32 actually reset/running, correct COM port,")
                print("   wiring, and that nothing else (e.g. idf.py monitor) has")
                print("   the port open.")
            else:
                print("Some text arrived but no DATA marker.")
                print("-> Check firmware logic / that uart_write_bytes is actually")
                print("   succeeding (uart_driver_install must be called first).")
            ser.close()
            sys.exit(1)

    if len(buffer) > 4096:
        buffer = buffer[-4096:]

print("\n" + "-" * 60)
print("Audio transmission started.")

# Keep anything after DATA\n
audio_start = buffer.find(b"DATA\n") + len(b"DATA\n")
audio_data = buffer[audio_start:]

print(f"Already received: {len(audio_data)} bytes")

# Receive exactly 5 seconds of PCM
while len(audio_data) < EXPECTED_BYTES:

    remaining = EXPECTED_BYTES - len(audio_data)

    chunk = ser.read(min(4096, remaining))

    if chunk:
        audio_data += chunk

    print(
        f"\rReceived {len(audio_data)}/{EXPECTED_BYTES} bytes",
        end=""
    )

print("\nRecording received.")

audio_data = audio_data[:EXPECTED_BYTES]

ser.close()

# Save WAV
with wave.open("mic_test.wav", "wb") as wav:
    wav.setnchannels(CHANNELS)
    wav.setsampwidth(SAMPLE_WIDTH)
    wav.setframerate(SAMPLE_RATE)
    wav.writeframes(audio_data)

print("Saved: mic_test.wav")