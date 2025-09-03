import os
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai

from app.services.emb_qdrant import search_similar

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

CHAT_MODEL = "models/gemini-1.5-flash"

SYSTEM_HINT = (
    "You are a helpful assistant. Answer ONLY using the provided context. "
    "If the answer is not in the context, say you don't know"
)

def build_context(hits) -> str:
    """
    Builds context string; if chunk_idx/doc_id present, show index and title.
    """
    lines = []
    for i, h in enumerate(hits, 1):
        txt = h.payload.get("text", "")
        meta = {k: v for k, v in h.payload.items() if k != "text"}
        lines.append(f"[{i}] {txt}\nMETA: {meta}")
    return "\n\n".join(lines)

def answer_with_rag(question: str, top_k: int = 5) -> dict:
    hits = search_similar(question, top_k=top_k)
    context = build_context(hits)

    prompt = (
        f"{SYSTEM_HINT}\n\n"
        f"Context: \n{context} \n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    model = genai.GenerativeModel(CHAT_MODEL)
    resp = model.generate_content(prompt)

    return {
        "answer": resp.text or "",
        "sources": [
            {
                "score": float(h.score),
                "text": h.payload.get("text", ""),
                "meta": {k: v for k, v in h.payload.items() if k != "text"}
            }
            for h in hits
        ],
    }
