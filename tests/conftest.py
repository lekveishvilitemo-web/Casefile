import os
import tempfile
from pathlib import Path

import pytest

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from app.config import Settings
from app.engine import Engine
from app.indexing.embeddings import HashingEmbedder
from app.rag.llm import LLMClient, LLMResult, ToolCall


class FakeLLM(LLMClient):
    provider = "fake"
    model = "fake-script"

    def complete(self, messages, *, tools=None, temperature=0.1, max_tokens=600):
        if tools:
            return LLMResult(
                content="",
                tool_calls=[ToolCall(id="call_finish", name="finish", arguments={})],
                provider=self.provider,
                model=self.model,
            )
        prompt = messages[-1]["content"]
        import re

        ids = re.findall(r"\[(doc_[a-z0-9]+_[a-z]\d+)\]", prompt, flags=re.I)
        citation = ids[0] if ids else "doc_missing_s0"
        return LLMResult(
            content=f"Answer: ledger-service v2.4.1\nThe deploy changed the retry count [{citation}].",
            provider=self.provider,
            model=self.model,
        )


@pytest.fixture()
def engine():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    settings = Settings(
        data_dir=root,
        corpus_dir=root / "corpus",
        chroma_dir=root / "chroma",
        upload_dir=root / "uploads",
        db_path=root / "casefile.db",
        model_path=root / "missing.gguf",
        embedding_model="hashing-test",
        llm_provider="fake",
        llm_model="fake-script",
        groq_api_key="",
        gemini_api_key="",
        openai_api_key="",
        openai_base_url="",
        chunk_strategy="sentence_pack",
        chunk_size=700,
        chunk_overlap=120,
        top_k=4,
        candidate_k=8,
        abstain_coverage=0.45,
        abstain_semantic=1.01,
        max_upload_bytes=1_000_000,
    )
    built = Engine(settings, HashingEmbedder(), FakeLLM())
    yield built
    built.store.close()
    tmp.cleanup()
