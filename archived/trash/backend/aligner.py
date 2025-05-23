# aligner.py
import os
import subprocess
from textgrid import TextGrid

def run_mfa(wav_path: str, transcript_path: str, lexicon_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "mfa", "align",
        wav_path,
        transcript_path,
        lexicon_path,
        output_dir,
        "--language", "kor",
        "--clean", "--overwrite"
    ]
    subprocess.run(cmd, check=True)

def extract_phones_from_textgrid(textgrid_path: str) -> list[str]:
    tg = TextGrid.fromFile(textgrid_path)
    tier = tg.getFirst("phones")
    phones = [entry.mark for entry in tier if entry.mark.strip()]
    return phones