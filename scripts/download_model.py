"""Download the local GGUF model used when no API key is configured."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(os.environ.get("LLM_MODEL_PATH", ROOT / "data" / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"))


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 1_000_000:
        print(f"Model already present at {DEST}")
        return
    path = hf_hub_download(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        local_dir=DEST.parent,
    )
    print(f"Model downloaded to {path}")


if __name__ == "__main__":
    main()
