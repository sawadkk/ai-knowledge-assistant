import re
from typing import List

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def sentence_split(text: str) -> List[str]:
    text = text.strip().replace("\r\n", "\n")
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return sents if sents else [text]

def combine_sents(
    sents: List[str],
    target_chars: int = 800,
    min_chars: int = 200,
    max_chars: int = 1000,
) -> List[str]:
    """
       Build chunks by combining sentences up to ~target_chars.
       Ensures chunks are not too tiny; caps at max_chars.
    """

    chunks: List[str] = []
    buf: List[str] = []
    size = 0

    for s in sents:
        s_len = len(s)
        if size + s_len + 1 <= max_chars:
            buf.append(s)
            size += s_len + 1
            continue
        if buf:
            chunk = " ".join(buf).strip()
            if len(chunk) >= min_chars or not chunks:
                chunks.append(chunk)
            else:
                chunks[-1] = (chunk[-1] + " " + chunk).strip()
    return chunks

def smart_chunks(
    text: str,
    target_chars: int = 800,
    min_chars: int = 200,
    max_chars: int = 1000,
) -> List[str]:
    t = text.strip()
    if len(t) <= min_chars:
        return [t]
    sents = sentence_split(t)
    return combine_sents(sents, target_chars, min_chars, max_chars)
