import os
import uuid
from typing import List
from dotenv import load_dotenv

import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
EMBED_MODEL = "models/text-embedding-004"
EMBED_DIM = 768

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
qdrant = QdrantClient(url=QDRANT_URL)


def ensure_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )

def _extract_embedding(res) -> list[float]:
    if isinstance(res, dict):
        if "embedding" in res:
            emb = res["embedding"]
            if isinstance(emb, dict) and "values" in emb:
                return emb["values"]
            if isinstance(emb, list):
                return emb
        if "embeddings" in res:
            embs = res["embeddings"]
            if isinstance(embs, list):
                if len(embs) == 0:
                    return []
                first = embs[0]
                if isinstance(first, dict) and "values" in first:
                    return first["values"]
                if isinstance(first, list):
                    return first
    try:
        return res.embedding.values
    except Exception:
        pass
    raise RuntimeError(f"Unexpected embedding response shape: {type(res)} -> {res}")

def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for t in texts:
        res = genai.embed_content(model=EMBED_MODEL, content=t)
        vectors.append(_extract_embedding(res))
    return vectors


def upsert_chunks(chunks: list[str], meta_list: list[dict]):
    ensure_collection()
    vectors = embed_texts(chunks)

    points = []
    for i, (vec, meta) in enumerate(zip(vectors, meta_list)):
        point_id = meta.get("id") or str(uuid.uuid4())

        points.append(
            PointStruct(
                id=point_id,
                vector=vec,
                payload={**meta, "text": chunks[i]},
            )
        )
    qdrant.upsert(collection_name=COLLECTION, points=points)

def search_similar(query: str, top_k: int = 5):
    q_vec = embed_texts([query])[0]
    result = qdrant.search(collection_name=COLLECTION, query_vector=q_vec, limit=top_k)
    return result
