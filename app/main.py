from fastapi import FastAPI
from app.api import routes_rag

app = FastAPI()
app.include_router(routes_rag.router, prefix="", tags=["rag"])