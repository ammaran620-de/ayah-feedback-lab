"""Run the existing ASR wording pipeline on the verifier's identical test set."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from align import AYAT, score
from app import infer
from train_verifier import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    with (args.run / "held_out_predictions.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    verdicts, any_differences, times = [], [], []
    examples = {}
    for index, row in enumerate(rows, 1):
        ayah = row["ayah"]
        transcript, signal, ms = infer((args.dataset / row["audio"]).read_bytes())
        result = score(ayah, transcript, signal)
        final_word = AYAT[ayah].split()[-1]
        final_flag = any(item["expected"] == final_word and item["label"] != "Matched" for item in result["rows"])
        verdicts.append("review" if result["status"] == "retry" else "flag" if final_flag else "no_flag")
        any_differences.append({"retry": "review", "difference": "flag", "match": "no_flag"}[result["status"]])
        times.append(ms)
        key = None
        if row["label"] == "1" and row["verdict"] == "flag" and result["status"] == "match":
            key = "whisper_wording_match_acoustic_flag"
        elif row["label"] == "0" and row["verdict"] == "flag":
            key = "acoustic_false_flag"
        if key and key not in examples:
            examples[key] = {**row, "whisper_status": result["status"], "whisper_transcript": transcript}
        if index % 25 == 0 or index == len(rows):
            print(f"Scored {index}/{len(rows)} ASR clips", flush=True)
    labels = np.array([int(row["label"]) for row in rows])
    baseline = {
        "final_word_proxy": summarize(labels, verdicts),
        "any_word_difference_proxy": summarize(labels, any_differences),
        "inference_ms": {"median": float(np.median(times)), "p95": float(np.percentile(times, 95))},
    }
    path = args.run / "evaluation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["whisper_baseline"] = baseline
    report["held_out_examples"] = examples
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(baseline, indent=2))


if __name__ == "__main__":
    main()
