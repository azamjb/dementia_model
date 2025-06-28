
import whisper
import os

model = whisper.load_model("base")


input_dir = "audio/no_dementia"
output_dir = "transcripts/no_dementia"
os.makedirs(output_dir, exist_ok=True)

# Loop through .wav files
for filename in os.listdir(input_dir):

    if filename.endswith(".wav"):

        audio_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".wav", ".txt"))

        print(f"Transcribing {filename}...")

        try:
            result = model.transcribe(audio_path)
            transcript = result["text"]

            with open(output_path, "w") as f:
                f.write(transcript)

            print(f"Saved transcript to {output_path}")
        except Exception as e:
            print(f"Failed to transcribe {filename}: {e}")
