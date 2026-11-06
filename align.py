"""Deterministic word alignment shared by the demo and checks."""

import unicodedata

AYAT = {
    "112:1": "قُلْ هُوَ ٱللَّهُ أَحَدٌ",
    "112:2": "ٱللَّهُ ٱلصَّمَدُ",
    "112:3": "لَمْ يَلِدْ وَلَمْ يُولَدْ",
    "112:4": "وَلَمْ يَكُن لَّهُۥ كُفُوًا أَحَدٌۢ",
}
ORTHOGRAPHY = str.maketrans("ٱأإآ", "اااا")
FIXTURES = [
    ("Correct wording", "112:1", "قل هو الله أحد", (0, 0, 0)),
    ("Skipped word", "112:1", "قل الله أحد", (0, 1, 0)),
    ("Substituted word", "112:3", "لم يلد ولم يلد", (1, 0, 0)),
    ("Repeated word", "112:1", "قل هو هو الله أحد", (0, 0, 1)),
    ("Diacritics and punctuation", "112:2", "اللَّهُ، الصَّمَدُ!", (0, 0, 0)),
    ("Alternate script marks", "112:4", "ولم يكن له كفوا أحد", (0, 0, 0)),
    ("Two edits", "112:1", "قل قل الله صمد", (2, 0, 0)),
    ("Final word missing", "112:3", "لم يلد ولم", (0, 1, 0)),
]


def normalize(text):
    text = unicodedata.normalize("NFKC", text).translate(ORTHOGRAPHY)
    letters = []
    for ch in text:
        if unicodedata.category(ch).startswith("M") or ch in "ـۥۦ":
            continue
        if unicodedata.category(ch).startswith("L"):
            letters.append(ch)
        elif unicodedata.category(ch) != "Cf":
            letters.append(" ")
    return "".join(letters).split()


def align(reference, hypothesis):
    """Levenshtein alignment with deterministic diagonal-first tie breaking."""
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    rows = []
    i, j = n, m
    while i or j:
        if i and j:
            differs = reference[i - 1] != hypothesis[j - 1]
            if dp[i][j] == dp[i - 1][j - 1] + differs:
                rows.append(("Substituted" if differs else "Matched", i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i and dp[i][j] == dp[i - 1][j] + 1:
            rows.append(("Skipped", i - 1, None))
            i -= 1
        else:
            rows.append(("Extra", None, j - 1))
            j -= 1
    return list(reversed(rows))


def score(ayah, transcript, signal=None):
    reference = AYAT[ayah]
    hypothesis = normalize(transcript)
    base = {"ayah": ayah, "reference": reference, "transcript": transcript, "signal": signal}
    if not hypothesis or (signal is not None and signal < -0.9):
        return {**base, "status": "retry", "rows": [], "edits": None, "distance": None}

    operations = align(normalize(reference), hypothesis)
    counts = {name: 0 for name in ("Substituted", "Skipped", "Extra")}
    rows = []
    display_words = reference.split()
    for label, i, j in operations:
        if label in counts:
            counts[label] += 1
        rows.append({
            "label": label,
            "expected": display_words[i] if i is not None else None,
            "heard": hypothesis[j] if j is not None else None,
        })
    total = sum(counts.values())
    return {
        **base,
        "status": "difference" if total else "match",
        "rows": rows,
        "edits": counts,
        "distance": total / len(display_words),
    }


def run_fixtures():
    results = []
    for name, ayah, transcript, expected in FIXTURES:
        result = score(ayah, transcript)
        actual = tuple(result["edits"].values())
        results.append({"name": name, "passed": actual == expected, "expected": expected, "actual": actual})
    for name, transcript, signal in [("Empty transcript", "", None), ("Weak ASR signal", "قل هو الله أحد", -1.2)]:
        results.append({"name": name, "passed": score("112:1", transcript, signal)["status"] == "retry"})
    return results
