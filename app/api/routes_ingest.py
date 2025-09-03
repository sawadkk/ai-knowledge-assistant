from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List
import uuid

from app.services.chunking import smart_chunks
from app.services.emb_qdrant import upsert_chunks

router = APIRouter()

class IngestTextIn(BaseModel):
    doc_id: Optional[str] = Field(default=None, description="Provide to overwrite/append; else auto-gen")
    title: Optional[str] = "untitled"
    text: str
    max_chars: int = 800

class IngestTextOut(BaseModel):
    doc_id: str
    chunks: int

@router.post("/ingest/text", response_model=IngestTextOut)
async def ingest_text(in_: IngestTextIn):
    doc_id = in_.doc_id or str(uuid.uuid4())
    chunks: List[str] = smart_chunks(
        in_.text, target_chars=in_.max_chars, min_chars=200, max_chars=max(400, in_.max_chars))
    metas = [{"doc_id": doc_id, "title": in_.title, "chunk_idx": i} for i in range(len(chunks))]
    upsert_chunks(chunks, metas)
    return IngestTextOut(doc_id=doc_id, chunks=len(chunks))
