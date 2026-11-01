import os, tempfile
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor

app = FastAPI(title="Chord Finder API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Loaded once at startup. To use a different model later, replace detect_chords().
features = CNNChordFeatureProcessor()
decoder = CRFChordRecognitionProcessor()

MAX_MB = 25


def detect_chords(wav_path):
    out = decoder(features(wav_path))
    return [{"start": round(float(s), 2), "end": round(float(e), 2), "chord": str(c)} for s, e, c in out]


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
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
        sf.write(wav, audio.mean(axis=1), sr)
        return {"chords": detect_chords(wav)}
