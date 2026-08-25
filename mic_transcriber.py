#!/usr/bin/python3
import os
import sys
import time
import threading

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "large-v3-turbo"

SAMPLE_RATE = 16_000
CHANNELS = 1

# Your NVIDIA GPU
DEVICE = "cuda"
COMPUTE_TYPE = "float16"


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

recording = False
audio_chunks = []

lock = threading.Lock()


# ---------------------------------------------------------------------------
# Audio callbacks
# ---------------------------------------------------------------------------

def audio_callback(indata, frames, time_info, status):
    """Receive audio chunks from the microphone."""

    if status:
        print(f"\nAudio status: {status}", file=sys.stderr)

    with lock:
        if recording:
            audio_chunks.append(indata.copy())


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def start_recording():
    """Start capturing microphone audio."""

    global recording, audio_chunks

    with lock:
        audio_chunks = []
        recording = True

    print("\n🎙️  Recording...", flush=True)


def stop_recording():
    """Stop capturing microphone audio and return the recording."""

    global recording

    with lock:
        recording = False

        if not audio_chunks:
            return None

        audio = np.concatenate(audio_chunks, axis=0)

    print("⏹️  Recording stopped.", flush=True)

    # Convert from (samples, 1) to (samples,)
    return audio[:, 0].astype(np.float32)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(model, audio):
    """Transcribe microphone audio and save the result."""

    if audio is None or len(audio) == 0:
        return

    duration = len(audio) / SAMPLE_RATE

    print(f"⏱️  Audio length: {duration:.1f}s")
    print("🧠 Transcribing...", flush=True)

    start = time.perf_counter()

    segments, info = model.transcribe(
        audio,
        language="pt",
        beam_size=5,
        vad_filter=True,
    )

    segments = list(segments)

    elapsed = time.perf_counter() - start

    # Create output directory
    output_dir = os.path.expanduser("~/transcriptions")
    os.makedirs(output_dir, exist_ok=True)

    # Timestamp-based filename
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(
        output_dir,
        f"transcription_{timestamp}.txt",
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Language: {info.language}\n")
        f.write(
            f"Language probability: "
            f"{info.language_probability:.2f}\n"
        )
        f.write(f"Audio length: {duration:.1f}s\n")
        f.write(f"Processing time: {elapsed:.2f}s\n")
        f.write("\n")
        f.write("=" * 70)
        f.write("\n\n")

        for segment in segments:
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)

            text = segment.text.strip()

            if text:
                f.write(
                    f"[{start_time} --> {end_time}] "
                    f"{text}\n"
                )

    print()
    print("=" * 70)

    for segment in segments:
        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)

        text = segment.text.strip()

        if text:
            print(
                f"[{start_time} --> {end_time}] "
                f"{text}"
            )

    print("=" * 70)

    print(f"Language: {info.language}")
    print(f"Language probability: {info.language_probability:.2f}")
    print(
        f"Processing time: {elapsed:.2f}s "
        f"({duration / elapsed:.1f}x realtime)"
    )

    print(f"\n💾 Saved: {output_file}")


# ---------------------------------------------------------------------------
# Timestamp formatting
# ---------------------------------------------------------------------------

def format_timestamp(seconds):
    """Convert seconds into HH:MM:SS."""

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global recording
    
    print("Loading Whisper model...")
    print()

    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )

    print("Model loaded successfully.")
    print()
    print("=" * 70)
    print("OFFLINE BRAZILIAN PORTUGUESE TRANSCRIBER")
    print("=" * 70)
    print()
    print("Hold SPACE to record.")
    print("Release SPACE to transcribe.")
    print("Press Q to quit.")
    print()

    # Open microphone stream
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=audio_callback,
        )
    except Exception as exc:
        print(f"Could not open microphone: {exc}")
        sys.exit(1)

    # We need keyboard input.
    # This uses the terminal in raw mode on Linux.
    import termios
    import tty
    import select

    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        with stream:
            while True:
                ready, _, _ = select.select(
                    [sys.stdin],
                    [],
                    [],
                    0.1,
                )

                if not ready:
                    continue

                key = sys.stdin.read(1)

                # Quit
                if key.lower() == "q":
                    break

                # Start recording
                if key == " " and not recording:
                    start_recording()

                    # Wait until SPACE is released.
                    while True:
                        ready, _, _ = select.select(
                            [sys.stdin],
                            [],
                            [],
                            0.05,
                        )

                        if not ready:
                            continue

                        released_key = sys.stdin.read(1)

                        if released_key == " ":
                            break

                        if released_key.lower() == "q":
                            recording = False
                            return

                    audio = stop_recording()

                    transcribe(model, audio)

                    print()
                    print("Hold SPACE to record again.")
                    print("Press Q to quit.")

    finally:
        termios.tcsetattr(
            sys.stdin,
            termios.TCSADRAIN,
            old_settings,
        )

    print("\nGoodbye.")


if __name__ == "__main__":
    main()
