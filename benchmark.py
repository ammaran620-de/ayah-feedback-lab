"""Evaluate a UTF-8 CSV of labeled recitation clips using the same pipeline as the demo."""

import argparse
from collections import Counter
import csv
from pathlib import Path
from statistics import median

from align import AYAT, align, normalize, score
from app import MAX_AUDIO, infer


def ratio(numerator, denominator):
    return f"{numerator / denominator:.1%} ({numerator}/{denominator})" if denominator else "n/a (0 clips)"


def edits(reference, hypothesis):
    """Count edit types at reference indices or insertion slots."""
    counts = Counter()
    slot = 0
    for kind, i, _ in align(reference, hypothesis):
        if kind == "Extra":
            counts[kind, slot] += 1
        else:
            if kind != "Matched":
                counts[kind, i] += 1
            slot = i + 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or not {"audio", "ayah", "spoken_words"} <= rows[0].keys():
        parser.error("CSV needs at least one row with audio, ayah, spoken_words columns.")

    total_edits = total_words = retries = 0
    correct = errors = false_flags = caught = 0
    times = []
    exact_hits = gold_edits = predicted_edits = 0
    for row in rows:
        audio_path = args.manifest.parent / row["audio"]
        ayah, spoken = row["ayah"].strip(), normalize(row["spoken_words"])
        if ayah not in AYAT or not spoken or not 0 < audio_path.stat().st_size <= MAX_AUDIO:
            parser.error(f"Invalid ayah, empty spoken_words or unsupported size for {audio_path}.")
        transcript, signal, elapsed = infer(audio_path.read_bytes())
        verdict = score(ayah, transcript, signal)["status"]
        word_edits = sum(label != "Matched" for label, _, _ in align(spoken, normalize(transcript)))
        total_edits += word_edits
        total_words += len(spoken)
        retries += verdict == "retry"
        times.append(elapsed)
        reference = normalize(AYAT[ayah])
        gold = edits(reference, spoken)
        predicted = edits(reference, normalize(transcript)) if verdict != "retry" else Counter()
        exact_hits += sum((gold & predicted).values())
        gold_edits += sum(gold.values())
        predicted_edits += sum(predicted.values())
        actual_error = bool(gold)
        if actual_error:
            errors += 1
            caught += bool(gold & predicted)
        else:
            correct += 1
            false_flags += verdict == "difference"
        print(f"{audio_path.name}  {ayah}  {'word error' if actual_error else 'correct'}  {verdict}  {elapsed} ms")

    print(f"ASR word error rate: {ratio(total_edits, total_words)}")
    print(f"False flag rate on correct clips: {ratio(false_flags, correct)}")
    print(f"Error clips with a matching edit location and type: {ratio(caught, errors)}")
    print(f"Edit location/type precision: {ratio(exact_hits, predicted_edits)}")
    print(f"Edit location/type recall: {ratio(exact_hits, gold_edits)}")
    print(f"Retry rate: {ratio(retries, len(rows))}")
    print(f"Median inference time: {median(times):.0f} ms (excludes decode and model load)")


if __name__ == "__main__":
    main()
