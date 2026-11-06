"""CPU inference for a scoped final-dal acoustic classifier."""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.fft import dct

MODEL_PATH = Path(__file__).with_name("verifier_model.json")
SUPPORTED = {"112:1", "112:2"}
SAMPLE_RATE = 16000
FRAME, HOP, FFT = 400, 160, 512
BINS = 24
MIN_FRAME_RMS = 0.008

def mel_filters():
    limits = 2595 * np.log10(1 + np.array([80, 7600]) / 700)
    hz = 700 * (10 ** (np.linspace(*limits, BINS + 2) / 2595) - 1)
    freq = np.fft.rfftfreq(FFT, 1 / SAMPLE_RATE)
    rise = (freq - hz[:-2, None]) / (hz[1:-1, None] - hz[:-2, None])
    fall = (hz[2:, None] - freq) / (hz[2:, None] - hz[1:-1, None])
    return np.maximum(0, np.minimum(rise, fall))

MEL = mel_filters()

def frame_levels(samples):
    frames = np.lib.stride_tricks.sliding_window_view(samples, FRAME)[::HOP]
    return frames, np.sqrt(np.mean(frames ** 2, axis=1))


def has_signal(samples):
    if len(samples) < FRAME or not np.isfinite(samples).all():
        return False
    _, rms = frame_levels(samples)
    return bool(np.max(rms) >= MIN_FRAME_RMS)


def acoustic_features(samples, crop="tail"):
    """Return spectral/temporal features, or None when audio quality is too weak."""
    if len(samples) < FRAME or not np.isfinite(samples).all():
        return None
    frames, rms = frame_levels(samples)
    peak = float(np.max(rms))
    if peak < MIN_FRAME_RMS:
        return None
    active = np.flatnonzero(rms >= peak * 0.045)
    if active[-1] - active[0] < 22:
        return None

    power = np.abs(np.fft.rfft(frames * np.hanning(FRAME), n=FFT)) ** 2
    bands = np.log(np.maximum(power @ MEL.T, 1e-10))
    mfcc = dct(bands, type=2, axis=1, norm="ortho")[:, 1:13]
    log_rms = np.log(np.maximum(rms / peak, 1e-5))
    crossings = np.mean(np.diff(frames > 0, axis=1), axis=1)
    track = np.column_stack((mfcc, log_rms, crossings))
    if crop == "head":
        track = track[active[0]:min(active[-1] + 1, active[0] + 60)]
    else:
        end = active[-1] + 1
        start = max(active[0], end - (36 if crop == "short" else 60))
        track = track[start:end]
    features = []
    for chunk in np.array_split(track, 3):
        features.extend(np.mean(chunk, axis=0))
        features.extend(np.std(chunk, axis=0))
    return np.asarray(features)


def error_score(features, model):
    z = np.dot((features - model["mean"]) / model["scale"], model["coef"]) + model["intercept"]
    return float(1 / (1 + np.exp(-np.clip(z, -50, 50))))


def decision(score, low, high):
    if score < low:
        return "no_flag"
    if score >= high:
        return "flag"
    return "review"


@lru_cache(maxsize=1)
def load_model():
    with MODEL_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def verify(samples, ayah):
    if ayah not in SUPPORTED or not MODEL_PATH.is_file():
        return None
    model = load_model()
    features = acoustic_features(samples, model["crop"])
    if features is None:
        return {"status": "review", "score": None}
    value = error_score(features, model)
    return {"status": decision(value, model["low"], model["high"]), "score": round(value, 3)}
