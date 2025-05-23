# main.py
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
import os
import shutil
from g2p import sentence_to_phonemes
from aligner import run_mfa, extract_phones_from_textgrid
from analyzer import compare_phonemes

app = FastAPI()

UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "data/results"
LEXICON_PATH = "data/lexicon.txt"
TRANSCRIPT_PATH = "data/reference.txt"

@app.post("/analyze")
async def analyze_pronunciation(file: UploadFile = File(...), text: str = Form(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # 1. Save audio file
    filename = os.path.splitext(file.filename)[0]
    wav_path = os.path.join(UPLOAD_DIR, filename + ".wav")
    with open(wav_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2. Generate lexicon & transcript
    phonemes = sentence_to_phonemes(text)
    with open(LEXICON_PATH, "w", encoding="utf-8") as f:
        f.write(f"{text}\t{' '.join(phonemes)}\n")
    with open(TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(f"{filename}|{text}\n")

    # 3. Run MFA
    run_mfa(UPLOAD_DIR, TRANSCRIPT_PATH, LEXICON_PATH, OUTPUT_DIR)

    # 4. Extract TextGrid
    tg_path = os.path.join(OUTPUT_DIR, filename + ".TextGrid")
    actual_phones = extract_phones_from_textgrid(tg_path)

    # 5. Compare
    result = compare_phonemes(phonemes, actual_phones)

    return JSONResponse({
        "original": text,
        "phonemes": result
    })