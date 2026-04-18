from __future__ import annotations

import os
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.acp_subprocess import run_acp_subprocess

ACP_MARKER_BASE_URL = "acp://gemini"
_DEFAULT_TIMEOUT_SECONDS = 900.0


def _resolve_command() -> str:
    return os.getenv("GEMINI_ACP_COMMAND", "").strip() or "gemini"


def _resolve_args() -> list[str]:
    raw = os.getenv("GEMINI_ACP_ARGS", "").strip()
    return shlex.split(raw) if raw else ["--output-format", "stream-json", "--approval-mode", "yolo"]


def _parse_gemini_event(evt: dict) -> tuple[str, bool]:
    """Parse one NDJSON event from the Gemini CLI stream-json output.

    Shapes seen in practice::

        {"type":"message","role":"assistant","content":"PONG"}
        {"type":"message","role":"assistant","content":[{"type":"text","text":"PONG"}]}
        {"type":"result","response":"PONG"}

    Returns ``(text, finished)``. A ``result`` event terminates the turn.
    """
    evt_type = evt.get("type")
    if evt_type == "result":
        val = evt.get("response", "")
        if isinstance(val, str) and val.strip():
            return val.strip(), True
        return "", True
    if evt_type == "error":
        msg = evt.get("error") or evt.get("message") or "unknown error"
        raise RuntimeError(f"Gemini CLI error: {msg}")
    if evt_type == "message":
        role = str(evt.get("role") or "").lower()
        if role not in ("", "assistant"):
            return "", False
        content = evt.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip(), False
        if isinstance(content, list):
            parts = [
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") in ("text", None)
            ]
            joined = "\n".join(p for p in parts if p)
            if joined:
                return joined, False
    return "", False


class _GemChatCompletions:
    def __init__(self, client: "GeminiClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _GemChatNamespace:
    def __init__(self, client: "GeminiClient"):
        self.completions = _GemChatCompletions(client)


class GeminiClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        acp_cwd: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "gemini-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = command or _resolve_command()
        self._args = list(args or _resolve_args())
        self._cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = _GemChatNamespace(self)
        self.is_closed = False

    def close(self) -> None:
        self.is_closed = True

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        **_: Any,
    ) -> Any:
        prompt = _messages_to_prompt(messages or [])

        # Gemini CLI reads the prompt from stdin when --prompt is empty.
        # Using stdin avoids Windows' 32 KiB CreateProcess command-line cap
        # and the 8 KiB cmd.exe shim cap.
        cli_args = list(self._args)
        if "--prompt" not in cli_args and "-p" not in cli_args:
            cli_args += ["--prompt", ""]
        if model and "--model" not in cli_args and "-m" not in cli_args:
            cli_args += ["--model", model]

        text = run_acp_subprocess(
            self._command,
            cli_args,
            prompt,
            parse_event=_parse_gemini_event,
            timeout_seconds=float(timeout or _DEFAULT_TIMEOUT_SECONDS),
            cwd=self._cwd,
            cli_name="gemini",
        )
        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        msg = SimpleNamespace(
            content=text,
            tool_calls=[],
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage, model=model or "gemini-acp")


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = str(m.get("role") or "user").strip().lower()
        content = m.get("content") or ""
        if isinstance(content, list):
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
            content = "\n".join(texts)
        label = {"system": "System", "user": "User", "assistant": "Assistant"}.get(role, role.title())
        if str(content).strip():
            parts.append(f"{label}:\n{content}")
    parts.append("Continue from the latest user message.")
    return "\n\n".join(parts)
