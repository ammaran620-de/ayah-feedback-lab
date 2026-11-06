"""Train, tune and evaluate the final-dal experiment. Audio stays in the dataset cache."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import re

from faster_whisper.audio import decode_audio
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from verifier import acoustic_features, decision, error_score

SEED = 2026
TARGET = "خطأ في قلقلة حرف الدال"
FILE_PATTERN = re.compile(r"data/ID(\d+)V([12])[TF]\.wav")


def candidates(root):
    with (root / "metadata.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            match = FILE_PATTERN.fullmatch(row["file_name"])
            if not match or match[2] != row["verse_number"]:
                continue
            if row["label_name"] == "correct":
                label = 0
            elif row["label_name"] == "error" and row["error_explanation"].strip() == TARGET:
                label = 1
            else:
                continue
            yield row["file_name"], match[1], match[2], label


def load_data(root):
    seen = defaultdict(list)
    metadata = list(candidates(root))
    unique = list(dict.fromkeys(metadata))
    for name, speaker, verse, label in unique:
        path = root / name
        samples = decode_audio(str(path))
        digest = sha256(np.round(samples * 32767).astype("<i2").tobytes()).hexdigest()
        seen[digest].append((name, speaker, verse, label, samples))
    data = []
    duplicates = weak = 0
    for cluster in seen.values():
        if len(cluster) > 1:
            duplicates += len(cluster)
            continue
        name, speaker, verse, label, samples = cluster[0]
        features = {crop: acoustic_features(samples, crop) for crop in ("tail", "short", "head")}
        weak += features["tail"] is None
        data.append((name, speaker, verse, label, features))
    return data, {
        "duplicate_metadata_rows_removed": len(metadata) - len(unique),
        "duplicate_pcm_rows_removed": duplicates,
        "low_signal_rows": weak,
    }


def fit(x, y, c):
    scaler = StandardScaler().fit(x)
    estimator = LogisticRegression(C=c, max_iter=2000, random_state=SEED).fit(scaler.transform(x), y)
    return {
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": estimator.coef_[0].tolist(), "intercept": float(estimator.intercept_[0]),
    }


def summarize(labels, verdicts):
    pairs = Counter(zip(labels.tolist(), verdicts))
    return {
        "correct_clips": int(sum(labels == 0)),
        "error_clips": int(sum(labels == 1)),
        "true_flags": pairs[1, "flag"],
        "false_flags": pairs[0, "flag"],
        "false_passes": pairs[1, "no_flag"],
        "correct_passes": pairs[0, "no_flag"],
        "review_correct": pairs[0, "review"],
        "review_errors": pairs[1, "review"],
    }


def choose_thresholds(labels, values):
    """Penalize false flags twice as much as misses and require 80% validation coverage."""
    selected = None
    for low in np.arange(.2, .81, .05):
        for high in np.arange(low, .91, .05):
            verdicts = [decision(value, low, high) if np.isfinite(value) else "review" for value in values]
            counts = summarize(labels, verdicts)
            review = counts["review_correct"] + counts["review_errors"]
            if review / len(labels) > .2:
                continue
            cost = 2 * counts["false_flags"] + counts["false_passes"] + .5 * review
            key = (cost, counts["false_flags"], review)
            if selected is None or key < selected[0]:
                selected = (key, float(low), float(high))
    if selected is None:
        return .5, .5
    return selected[1:]


def train(root, output):
    data, audit = load_data(root)
    y = np.array([row[3] for row in data])
    groups = np.array([row[1] for row in data])
    strata = [row[2] + str(row[3]) for row in data]
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(y, strata, groups))
    test = folds[0][1]
    valid = folds[1][1]
    training = np.setdiff1d(np.arange(len(data)), np.r_[test, valid])
    splits = {"train": training, "validation": valid, "test": test}
    assert sum(len(set(groups[idx])) for idx in splits.values()) == len(set(groups))

    def features(idx, crop):
        return [data[i][4][crop] for i in idx]

    def probabilities(idx, crop, model):
        return np.asarray([error_score(x, model) if x is not None else np.nan for x in features(idx, crop)])

    def auc(labels, values):
        mask = np.isfinite(values)
        return float(roc_auc_score(labels[mask], values[mask]))

    def fit_crop(crop, c):
        train_features = features(training, crop)
        mask = [x is not None for x in train_features]
        return fit(np.asarray([x for x in train_features if x is not None]), y[training][mask], c)

    runs = []
    for crop in ("tail", "short"):
        for c in (.01, .1, 1.):
            model = fit_crop(crop, c)
            values = probabilities(valid, crop, model)
            runs.append((auc(y[valid], values), crop, c, model))
    selected_auc, crop, c, model = max(runs, key=lambda run: run[0])
    low, high = choose_thresholds(y[valid], probabilities(valid, crop, model))
    model.update({"crop": crop, "low": low, "high": high, "C": c, "schema": 1})
    values = probabilities(test, crop, model)
    verdicts = [decision(value, low, high) if np.isfinite(value) else "review" for value in values]
    head = fit_crop("head", c)
    result = {
        "seed": SEED,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "dataset_metadata_sha256": sha256((root / "metadata.csv").read_bytes()).hexdigest(),
        "data_audit": audit,
        "splits": {key: {"clips": len(idx), "participant_ids": len(set(groups[idx])),
                         "error_clips": int(y[idx].sum())} for key, idx in splits.items()},
        "validation_runs": [{"crop": crop, "C": c, "auc": score} for score, crop, c, _ in runs],
        "selected_validation_auc": selected_auc,
        "test_auc": auc(y[test], values),
        "test_head_control_auc": auc(y[test], probabilities(test, "head", head)),
        "test": summarize(y[test], verdicts),
        "test_by_ayah": {verse: summarize(y[test][mask], np.asarray(verdicts)[mask]) for verse in ("1", "2")
                         for mask in [[data[i][2] == verse for i in test]]},
    }
    output.mkdir(exist_ok=True, parents=True)
    for filename, contents in [("verifier_model.json", model), ("evaluation.json", result)]:
        (output / filename).write_text(json.dumps(contents, indent=2) + "\n", encoding="utf-8")
    with (output / "held_out_predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["audio", "ayah", "label", "acoustic_score", "verdict"])
        for index, value, verdict in zip(test, values, verdicts):
            writer.writerow([data[index][0], "112:" + data[index][2], y[index], value, verdict])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    train(args.dataset, args.output)
