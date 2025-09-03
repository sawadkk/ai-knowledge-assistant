import os
from dotenv import load_dotenv

load_dotenv()

from app.services.emb_qdrant import upsert_chunks, search_similar

chunks = [
    "Django and DRF are used to build REST APIs.",
    "Celery with Redis handles long running tasks.",
    "Qdrant stores embeddings for semantic search."
]
metas = [{"doc_id": "demo", "chunk_idx": i} for i in range(len(chunks))]

upsert_chunks(chunks, metas)

hits = search_similar("Which DB helps with semantic search?", top_k=3)
for h in hits:
    print(round(h.score, 4), "→", h.payload["text"])
