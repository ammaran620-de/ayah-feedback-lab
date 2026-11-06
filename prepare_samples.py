"""Fetch one public recitation and make clearly labeled audio demo cases."""

import argparse
import io
import json
from pathlib import Path
from urllib.request import urlopen
import wave

from faster_whisper.audio import decode_audio
import numpy as np

SOURCE = "https://everyayah.com/data/Alafasy_128kbps/112001.mp3"


def write_wav(path, samples):
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as file:
        file.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        file.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out", action="store_true", help="Download labeled examples from the recorded evaluation")
    args = parser.parse_args()
    output = Path("demo-clips")
    output.mkdir(exist_ok=True)
    if args.held_out:
        from download_dataset import BASE
        report = Path(__file__).with_name("evaluation.json")
        examples = json.loads(report.read_text(encoding="utf-8"))["held_out_examples"]
        for label, example in examples.items():
            path = output / f"{label}-{Path(example['audio']).name}"
            with urlopen(BASE + example["audio"], timeout=35) as response:
                path.write_bytes(response.read())
            print(f"{path} — select ayah {example['ayah']}")
        return
    with urlopen(SOURCE, timeout=30) as response:
        mp3 = response.read()
    audio = decode_audio(io.BytesIO(mp3), sampling_rate=16000)
    (output / "112-1-original-alafasy.mp3").write_bytes(mp3)
    # Remove the approximate 0.46–0.82 s window for هُوَ from this specific recording.
    write_wav(output / "112-1-synthetic-skipped-word.wav", np.concatenate((audio[:7360], audio[13120:])))
    write_wav(output / "synthetic-silence.wav", np.zeros(16000))
    print(f"Ready: {output.resolve()}")


if __name__ == "__main__":
    main()
