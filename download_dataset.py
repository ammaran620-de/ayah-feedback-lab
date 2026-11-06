"""Fetch the pinned dataset metadata and only the WAV files used by the experiment."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


REVISION = "973b5ef1d1532bcbe7e9a5a5b0dce125cc4d0dac"
BASE = f"https://huggingface.co/datasets/MuazAhmad7/Surah_Ikhlas-Labeled_Dataset/resolve/{REVISION}/"


def download(root, name):
    target = root / name
    if target.is_file():
        return
    partial = target.with_suffix(".part")
    for attempt in range(3):
        try:
            with urlopen(BASE + name, timeout=35) as response:
                partial.write_bytes(response.read())
            partial.replace(target)
            return
        except (OSError, URLError):
            if attempt == 2:
                raise


def main():
    from train_verifier import candidates

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset-cache"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = args.output
    (root / "data").mkdir(parents=True, exist_ok=True)
    download(root, "metadata.csv")
    names = sorted({name for name, *_ in candidates(root)})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, _ in enumerate(pool.map(lambda name: download(root, name), names), 1):
            if index % 50 == 0 or index == len(names):
                print(f"{index}/{len(names)} clips ready", flush=True)


if __name__ == "__main__":
    main()
