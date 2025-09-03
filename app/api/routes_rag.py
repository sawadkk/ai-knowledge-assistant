from fastapi import APIRouter
from pydantic import BaseModel
from app.services.rag_answer import answer_with_rag

router = APIRouter()

class RagIn(BaseModel):
    question: str
    top_k: int = 5

class RagOut(BaseModel):
    answer: str
    sources: list

@router.post("/rag", response_model=RagOut)
async def rag_endpoint(in_: RagIn):
    res = answer_with_rag(in_.question, top_k=in_.top_k)
    return RagOut(**res)

