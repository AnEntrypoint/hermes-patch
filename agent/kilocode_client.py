from __future__ import annotations

import os
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.acp_subprocess import (
    ensure_hermes_llm_agent,
    parse_opencode_style,
    run_acp_subprocess,
)

ACP_MARKER_BASE_URL = "acp://kilocode"
_DEFAULT_TIMEOUT_SECONDS = 900.0


def _resolve_command() -> str:
    return os.getenv("KILOCODE_ACP_COMMAND", "").strip() or "kilo"


def _resolve_args() -> list[str]:
    raw = os.getenv("KILOCODE_ACP_ARGS", "").strip()
    return shlex.split(raw) if raw else []


def _kilo_config_dir() -> Path:
    return Path(os.path.expanduser("~/.config/kilo"))


class _KCChatCompletions:
    def __init__(self, client: "KiloCodeClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _KCChatNamespace:
    def __init__(self, client: "KiloCodeClient"):
        self.completions = _KCChatCompletions(client)


class KiloCodeClient:
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
        self.api_key = api_key or "kilocode-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = command or _resolve_command()
        self._args = list(args or _resolve_args())
        self._cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = _KCChatNamespace(self)
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
        ensure_hermes_llm_agent(_kilo_config_dir())
        prompt = _messages_to_prompt(messages or [])

        cli_args = list(self._args) + ["run", "--format", "json"]
        # Use the zero-tool hermes-llm agent by default so kilo performs a
        # single LLM completion and exits on step-finish instead of looping
        # through its own tool-calling turns.
        if "--agent" not in cli_args:
            cli_args += ["--agent", "hermes-llm"]
        if model:
            cli_args += ["--model", model]

        text = run_acp_subprocess(
            self._command,
            cli_args,
            prompt,
            parse_event=parse_opencode_style,
            timeout_seconds=float(timeout or _DEFAULT_TIMEOUT_SECONDS),
            cwd=self._cwd,
            cli_name="kilo",
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
        return SimpleNamespace(choices=[choice], usage=usage, model=model or "kilocode-acp")


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
