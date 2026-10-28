"""HTTP API and the Casefile interface."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import __version__
from app.engine import build_engine, document_dict
from app.ingestion.loaders import UnsupportedFormat

STATIC_DIR = Path(__file__).resolve().parent / "static"
EVAL_PATH = Path(os.environ.get("EVAL_SUMMARY_PATH", "data/evaluation_summary.json"))

app = FastAPI(title="Casefile", version=__version__)
engine = None


def get_engine():
    global engine
    if engine is None:
        engine = build_engine()
    return engine


class TextIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    source: str = "paste"


class QuestionIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    chunk_strategy: str | None = None
    retrieval: str = "hybrid"


@app.exception_handler(UnsupportedFormat)
async def unsupported_format(_, exc: UnsupportedFormat):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/api/health")
def health():
    stats = get_engine().stats()
    return {"status": "ok", "version": __version__, "documents": stats["documents"]}


@app.get("/api/stats")
def stats():
    return get_engine().stats()


@app.get("/api/documents")
def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    q: str | None = None,
):
    rows = get_engine().store.list_documents(limit=limit, offset=offset, query=q)
    return {
        "total": get_engine().store.count_documents(q),
        "limit": limit,
        "offset": offset,
        "documents": [document_dict(row) for row in rows],
    }


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    document = get_engine().store.get_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document_dict(document, include_text=True)


@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    force: str = Form(default="false"),
):
    data = await _read_upload(file)
    force_flag = force.lower() in {"1", "true", "yes", "on"}
    try:
        outcomes = get_engine().add_upload(file.filename or "upload.txt", data, force=force_flag)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"results": outcomes}


@app.post("/api/documents/text")
def paste_document(body: TextIn):
    try:
        return get_engine().add_text(body.title, body.text, source=body.source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/documents/{doc_id}")
async def replace_document(doc_id: str, file: UploadFile = File(...)):
    if get_engine().store.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    data = await _read_upload(file)
    try:
        return get_engine().update_upload(doc_id, file.filename or "upload.txt", data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    if not get_engine().delete(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": doc_id}


@app.post("/api/investigate")
def investigate(body: QuestionIn):
    try:
        return get_engine().investigate(
            body.question,
            strategy=body.chunk_strategy,
            retrieval=body.retrieval,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/evaluation")
def evaluation():
    if not EVAL_PATH.exists():
        return {"available": False, "detail": "No evaluation report has been generated yet."}
    return {"available": True, **json.loads(EVAL_PATH.read_text(encoding="utf-8"))}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > get_engine().settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds the upload limit.")
    return data
