"""Probe held-out audio under quieter input, noise, and clipped endings."""

import argparse
import csv
import json
from pathlib import Path

from faster_whisper.audio import decode_audio
import numpy as np

from train_verifier import summarize
from verifier import acoustic_features, decision, error_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    model = json.loads((args.run / "verifier_model.json").read_text(encoding="utf-8"))
    with (args.run / "held_out_predictions.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    rng = np.random.default_rng(2026)
    conditions = {"gain_minus_12_db": [], "white_noise_12_db_snr": [], "active_end_minus_120_ms": []}
    for row in rows:
        audio = decode_audio(str(args.dataset / row["audio"]))
        noise = rng.standard_normal(len(audio)) * np.sqrt(np.mean(audio**2)) * 10 ** (-12 / 20)
        active = np.flatnonzero(np.abs(audio) >= np.max(np.abs(audio)) * .025)
        truncated = audio[:max(0, int(active[-1]) - 1920)] if active.size else audio[:0]
        transformed = (audio * 10 ** (-12 / 20), np.clip(audio + noise, -1, 1), truncated)
        for verdicts, samples in zip(conditions.values(), transformed):
            features = acoustic_features(samples, model["crop"])
            value = error_score(features, model) if features is not None else None
            verdicts.append(decision(value, model["low"], model["high"]) if value is not None else "review")
    labels = np.array([int(row["label"]) for row in rows])
    result = {condition: summarize(labels, verdicts) for condition, verdicts in conditions.items()}
    path = args.run / "evaluation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["stress_tests"] = result
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
