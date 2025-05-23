# g2p.py
from g2pk import G2p

g2p = G2p()

def sentence_to_phonemes(text: str) -> list[str]:
    raw = g2p(text)
    return list(raw.replace(" ",""))