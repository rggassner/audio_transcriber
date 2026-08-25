from faster_whisper import WhisperModel

model = WhisperModel(
    "large-v3-turbo",
    device="cuda",
    compute_type="float16",
)

segments, info = model.transcribe(
    "file.wav",
    language="pt",
    beam_size=5,
)

with open("file.txt", "w", encoding="utf-8") as f:
    for segment in segments:
        f.write(segment.text.strip() + "\n")

print(f"Detected language: {info.language}")
print(f"Language probability: {info.language_probability:.2f}")
print("Done.")
