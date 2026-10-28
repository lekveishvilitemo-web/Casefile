"""LLM providers.

`auto` picks the first configured backend: Groq, Gemini, OpenAI-compatible,
then a local GGUF model. Network and model failures raise LLMError so the
agent can abstain or fall back without inventing a completion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx


class LLMError(RuntimeError):
    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider: str = ""
    model: str = ""


class LLMClient:
    provider = "none"
    model = ""

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 600,
    ) -> LLMResult:
        raise LLMError("No language model is configured.", "unavailable")


class OpenAICompatibleClient(LLMClient):
    def __init__(self, *, provider: str, model: str, api_key: str, base_url: str) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 600,
    ) -> LLMResult:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=60.0,
            )
        except httpx.TimeoutException as exc:
            raise LLMError("The language model timed out.", "timeout") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"The language model request failed: {exc}", "network") from exc
        if response.status_code == 429:
            raise LLMError("The language model rate limit was hit.", "rate_limit")
        if response.status_code in {401, 403}:
            raise LLMError("The language model rejected the API key.", "auth")
        if response.status_code >= 400:
            raise LLMError(
                f"The language model returned HTTP {response.status_code}.",
                "bad_response",
            )
        try:
            body = response.json()
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError("The language model returned an unexpected payload.", "bad_response") from exc
        tool_calls = []
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            arguments = function.get("arguments") or "{}"
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            tool_calls.append(
                ToolCall(
                    id=call.get("id") or f"call_{index}",
                    name=function.get("name") or "",
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )
        return LLMResult(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            provider=self.provider,
            model=self.model,
        )


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self.provider = "gemini"
        self.model = model
        self.api_key = api_key

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 600,
    ) -> LLMResult:
        contents = []
        system_bits = []
        for message in messages:
            role = message["role"]
            content = message.get("content") or ""
            if role == "system":
                system_bits.append(content)
                continue
            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": content}],
                }
            )
        if tools:
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "If you need a tool, reply with only JSON "
                                '{"tool":"name","arguments":{...}}. Tools: '
                                + ", ".join(tool["function"]["name"] for tool in tools)
                            )
                        }
                    ],
                }
            )
        payload: dict = {
            "contents": contents or [{"role": "user", "parts": [{"text": "Answer."}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system_bits:
            payload["systemInstruction"] = {"parts": [{"text": "\n".join(system_bits)}]}
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        try:
            response = httpx.post(url, json=payload, timeout=60.0)
        except httpx.TimeoutException as exc:
            raise LLMError("Gemini timed out.", "timeout") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Gemini request failed: {exc}", "network") from exc
        if response.status_code == 429:
            raise LLMError("Gemini rate limit was hit.", "rate_limit")
        if response.status_code in {401, 403}:
            raise LLMError("Gemini rejected the API key.", "auth")
        if response.status_code >= 400:
            raise LLMError(f"Gemini returned HTTP {response.status_code}.", "bad_response")
        try:
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError("Gemini returned an unexpected payload.", "bad_response") from exc
        tool_calls = _tool_calls_from_text(text)
        return LLMResult(
            content="" if tool_calls else text,
            tool_calls=tool_calls,
            provider=self.provider,
            model=self.model,
        )


class LocalLlamaClient(LLMClient):
    def __init__(self, model_path: str) -> None:
        self.provider = "local"
        self.model = model_path
        self.model_path = model_path
        self._llm = None

    def _load(self):
        if self._llm is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise LLMError("llama-cpp-python is not installed.", "unavailable") from exc
            try:
                self._llm = Llama(
                    model_path=self.model_path,
                    n_ctx=4096,
                    n_threads=4,
                    verbose=False,
                )
            except Exception as exc:
                raise LLMError(f"The local model failed to load: {exc}", "unavailable") from exc
        return self._llm

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.1,
        max_tokens: int = 600,
    ) -> LLMResult:
        llm = self._load()
        prompt_messages = list(messages)
        if tools:
            names = ", ".join(tool["function"]["name"] for tool in tools)
            prompt_messages = prompt_messages + [
                {
                    "role": "user",
                    "content": (
                        "If a tool would help, reply with only JSON "
                        '{"tool":"name","arguments":{...}}. '
                        f"Available tools: {names}. Otherwise answer normally."
                    ),
                }
            ]
        try:
            result = llm.create_chat_completion(
                messages=prompt_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = result["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            raise LLMError(f"Local generation failed: {exc}", "bad_response") from exc
        tool_calls = _tool_calls_from_text(text) if tools else []
        return LLMResult(
            content="" if tool_calls else text,
            tool_calls=tool_calls,
            provider=self.provider,
            model=self.model,
        )


def _tool_calls_from_text(text: str) -> list[ToolCall]:
    stripped = text.strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            return []
        stripped = stripped[start : end + 1]
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return []
    name = payload.get("tool") or payload.get("name")
    arguments = payload.get("arguments") or payload.get("parameters") or {}
    if not name or not isinstance(arguments, dict):
        return []
    return [ToolCall(id="call_text", name=str(name), arguments=arguments)]


def build_llm(settings) -> LLMClient:
    provider = settings.llm_provider.lower().strip()
    if provider in {"", "auto"}:
        if settings.groq_api_key:
            provider = "groq"
        elif settings.gemini_api_key:
            provider = "gemini"
        elif settings.openai_api_key:
            provider = "openai"
        elif settings.model_path.exists():
            provider = "local"
        else:
            provider = "none"
    if provider == "groq":
        if not settings.groq_api_key:
            raise LLMError("GROQ_API_KEY is not set.", "auth")
        return OpenAICompatibleClient(
            provider="groq",
            model=settings.llm_model or "llama-3.1-8b-instant",
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set.", "auth")
        return OpenAICompatibleClient(
            provider="openai",
            model=settings.llm_model or "gpt-4o-mini",
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set.", "auth")
        return GeminiClient(settings.gemini_api_key, settings.llm_model or "gemini-2.0-flash")
    if provider == "local":
        if not settings.model_path.exists():
            raise LLMError(
                f"Local model file was not found at {settings.model_path}.",
                "unavailable",
            )
        return LocalLlamaClient(str(settings.model_path))
    return LLMClient()
