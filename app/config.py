"""Runtime configuration. Secrets come from the environment, never from the repo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    corpus_dir: Path
    chroma_dir: Path
    upload_dir: Path
    db_path: Path
    model_path: Path
    embedding_model: str
    llm_provider: str
    llm_model: str
    groq_api_key: str
    gemini_api_key: str
    openai_api_key: str
    openai_base_url: str
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    candidate_k: int
    abstain_coverage: float
    abstain_semantic: float
    max_upload_bytes: int

    @classmethod
    def from_env(cls) -> "Settings":
        data = _path("DATA_DIR", "data")
        return cls(
            data_dir=data,
            corpus_dir=data / "corpus",
            chroma_dir=data / "chroma",
            upload_dir=data / "uploads",
            db_path=data / "casefile.db",
            model_path=_path(
                "LLM_MODEL_PATH",
                "data/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            ),
            embedding_model=os.environ.get(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            llm_provider=os.environ.get("LLM_PROVIDER", "auto"),
            llm_model=os.environ.get("LLM_MODEL", ""),
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            gemini_api_key=os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", ""),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            openai_base_url=os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ),
            chunk_strategy=os.environ.get("CHUNK_STRATEGY", "sentence_pack"),
            chunk_size=int(os.environ.get("CHUNK_SIZE", "700")),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "120")),
            top_k=int(os.environ.get("TOP_K", "6")),
            candidate_k=int(os.environ.get("CANDIDATE_K", "30")),
            abstain_coverage=float(os.environ.get("ABSTAIN_COVERAGE", "0.34")),
            abstain_semantic=float(os.environ.get("ABSTAIN_SEMANTIC", "0.28")),
            max_upload_bytes=int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))),
        )

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.corpus_dir,
            self.chroma_dir,
            self.upload_dir,
            self.model_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)
