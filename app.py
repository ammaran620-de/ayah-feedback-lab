"""Local interview demo. Run with Python 3.10+."""

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
from time import perf_counter

from align import AYAT, FIXTURES, run_fixtures, score

WEB = Path(__file__).parent / "web"
MODEL_ID = "OdyAsh/faster-whisper-base-ar-quran"
MODEL_REVISION = "8578bafd427d122f671b187706650e76edf6f86f"
MAX_AUDIO = 8 * 1024 * 1024
MODEL = None
MODEL_LOCK = threading.Lock()


def get_model(*, allow_download=False):
    global MODEL
    if MODEL is None:
        from faster_whisper import WhisperModel
        MODEL = WhisperModel(
            MODEL_ID, device="cpu", compute_type="int8", revision=MODEL_REVISION,
            local_files_only=not allow_download,
        )
    return MODEL


def decode_clip(audio):
    from faster_whisper.audio import decode_audio

    # Audio bytes are written to an unnamed/temporary file and never retained.
    with tempfile.TemporaryFile() as file:
        file.write(audio)
        file.seek(0)
        samples = decode_audio(file, sampling_rate=16000)
    duration = len(samples) / 16000
    if not 0.4 <= duration <= 20:
        raise ValueError("Choose an audio clip between 0.4 and 20 seconds.")

    return samples


def infer(audio):
    return transcribe(decode_clip(audio))


def transcribe(samples):
    with MODEL_LOCK:
        from huggingface_hub.errors import LocalEntryNotFoundError
        try:
            model = get_model()
        except LocalEntryNotFoundError as exc:
            raise RuntimeError("Model is missing. Run app.py --warmup first.") from exc
        started = perf_counter()
        # Keep the expected ayah out of the decoder so it cannot simply echo the reference.
        segments, _ = model.transcribe(
            samples, language="ar", task="transcribe", beam_size=3,
            condition_on_previous_text=False, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 250},
        )
        segments = [s for s in segments if s.text.strip()]
        elapsed_ms = round((perf_counter() - started) * 1000)
    transcript = " ".join(s.text.strip() for s in segments)
    weakest = min((s.avg_logprob for s in segments), default=None)
    return transcript, weakest, elapsed_ms


class Handler(BaseHTTPRequestHandler):
    def send_data(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data, status=200):
        self.send_data(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path == "/api/config":
            self.send_json({"ayat": AYAT, "fixtures": FIXTURES, "model": MODEL_ID})
            return
        if self.path == "/api/experiment":
            experiment = Path(__file__).with_name("evaluation.json")
            self.send_json(json.loads(experiment.read_text(encoding="utf-8")) if experiment.is_file() else {})
            return
        if self.path == "/api/checks":
            self.send_json(run_fixtures())
            return
        files = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
        if self.path not in files:
            self.send_error(404)
            return
        path = WEB / files[self.path]
        self.send_data(200, path.read_bytes(), mimetypes.guess_type(path)[0] + "; charset=utf-8")

    def do_POST(self):
        route, _, ayah = self.path.rpartition("/")
        if route not in ("/api/text", "/api/audio") or ayah not in AYAT:
            self.send_json({"error": "Select a supported ayah."}, 404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json({"error": "Invalid request size."}, 400)
            return
        limit = 4096 if route == "/api/text" else MAX_AUDIO
        if not 0 < size <= limit:
            self.send_json({"error": "Request is empty or too large. Audio limit: 8 MB."}, 400)
            return
        body = self.rfile.read(size)
        if route == "/api/text":
            try:
                transcript = body.decode()
            except UnicodeDecodeError:
                self.send_json({"error": "Transcript must be UTF-8."}, 400)
                return
            self.send_json({**score(ayah, transcript), "mode": "Transcript only"})
            return
        try:
            from verifier import SUPPORTED, has_signal, verify
            samples = decode_clip(body)
            if has_signal(samples):
                transcript, signal, elapsed = transcribe(samples)
                acoustic = verify(samples, ayah)
            else:
                transcript, signal, elapsed = "", None, 0
                acoustic = {"status": "review", "score": None} if ayah in SUPPORTED else None
        except ImportError:
            self.send_json({"error": "Audio dependencies are missing. Follow the audio setup in README.md."}, 503)
            return
        except ValueError as exc:
            self.send_json({"error": str(exc)}, 400)
            return
        except Exception as exc:
            print(f"Audio error: {type(exc).__name__}: {exc}", flush=True)
            self.send_json({"error": "Audio could not be analyzed. Try a supported clip and check model setup in README.md."}, 422)
            return
        self.send_json({**score(ayah, transcript, signal), "mode": "Local speech recognition", "elapsed_ms": elapsed, "acoustic": acoustic})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", action="store_true", help="Download/cache and load the model, then exit")
    parser.add_argument("--check", action="store_true", help="Run transcript and retry fixtures, then exit")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.check:
        results = run_fixtures()
        for item in results:
            print(f"{'PASS' if item['passed'] else 'FAIL'}  {item['name']}")
        raise SystemExit(0 if all(item["passed"] for item in results) else 1)
    if args.warmup:
        print(f"Loading {MODEL_ID}...", flush=True)
        get_model(allow_download=True)
        print("Model cached. Start app.py and analyze one clip before your interview.")
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Open http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
