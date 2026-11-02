import json, os, re, tempfile
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor

app = FastAPI(title="Chord Finder API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Loaded once at startup. To use a different model later, replace detect_chords().
features = CNNChordFeatureProcessor()
decoder = CRFChordRecognitionProcessor()

MAX_MB = 25

# ---------------------------------------------------------------- shared library
# Chords are cached by a SHA-256 of the audio the browser sends, so the same file
# is only ever analyzed once for everyone. Audio itself is never stored.
# Set DATABASE_URL to make this survive restarts; without it the cache is
# in-memory and the free Render instance loses it whenever it sleeps.

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_memory = {}


def _connect():
    import psycopg

    # prepare_threshold=None keeps this working through Supabase's transaction pooler.
    return psycopg.connect(DATABASE_URL, connect_timeout=5, prepare_threshold=None)


def init_cache():
    if not DATABASE_URL:
        print("chord cache: DATABASE_URL not set, using in-memory cache")
        return
    try:
        with _connect() as c:
            c.execute(
                "create table if not exists chord_cache ("
                "  sha text primary key,"
                "  chords jsonb not null,"
                "  created_at timestamptz not null default now())"
            )
        print("chord cache: connected")
    except Exception as e:
        print("chord cache: init failed, falling back to memory:", e)


def cache_get(sha):
    if not DATABASE_URL:
        return _memory.get(sha)
    try:
        with _connect() as c:
            row = c.execute("select chords from chord_cache where sha = %s", (sha,)).fetchone()
            return row[0] if row else None
    except Exception as e:
        print("chord cache: read failed:", e)
        return None


def cache_put(sha, chords):
    if not DATABASE_URL:
        _memory[sha] = chords
        return
    try:
        with _connect() as c:
            c.execute(
                "insert into chord_cache (sha, chords) values (%s, %s) on conflict (sha) do nothing",
                (sha, json.dumps(chords)),
            )
    except Exception as e:
        print("chord cache: write failed:", e)


def cache_count():
    if not DATABASE_URL:
        return len(_memory)
    try:
        with _connect() as c:
            return c.execute("select count(*) from chord_cache").fetchone()[0]
    except Exception as e:
        print("chord cache: count failed:", e)
        return 0


# ---------------------------------------------------------------- chord engine


def detect_chords(wav_path):
    out = decoder(features(wav_path))
    return [{"start": round(float(s), 2), "end": round(float(e), 2), "chord": str(c)} for s, e, c in out]


# ---------------------------------------------------------------- routes


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "shared_library": bool(DATABASE_URL), "songs": cache_count()}


@app.get("/chords/{sha}")
def cached_chords(sha: str):
    """Ask whether this exact audio has been analyzed before, without uploading it."""
    if not SHA_RE.match(sha):
        raise HTTPException(400, "Bad fingerprint.")
    chords = cache_get(sha)
    if chords is None:
        raise HTTPException(404, "Not analyzed yet.")
    return {"chords": chords, "cached": True}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), sha: str = Form("")):
    sha = sha.strip().lower()
    keyed = bool(SHA_RE.match(sha))

    if keyed:
        hit = cache_get(sha)
        if hit is not None:
            return {"chords": hit, "cached": True}

    data = await file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_MB} MB).")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in")
        wav = os.path.join(tmp, "in.wav")
        with open(src, "wb") as f:
            f.write(data)
        try:
            audio, sr = sf.read(src, dtype="float32", always_2d=True)
        except Exception:
            raise HTTPException(400, "Couldn't read that file. Use MP3, WAV, FLAC or OGG.")
        mono = audio.mean(axis=1)
        if sr != 44100:
            g = gcd(44100, int(sr))
            mono = resample_poly(mono, 44100 // g, int(sr) // g).astype("float32")
        sf.write(wav, mono, 44100)
        chords = detect_chords(wav)

    if keyed:
        cache_put(sha, chords)
    return {"chords": chords, "cached": False}


init_cache()
