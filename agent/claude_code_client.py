from __future__ import annotations

import os
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.acp_subprocess import run_acp_subprocess

ACP_MARKER_BASE_URL = "acp://claude-code"
_DEFAULT_TIMEOUT_SECONDS = 900.0


def _resolve_command() -> str:
    return os.getenv("CLAUDE_CODE_ACP_COMMAND", "").strip() or "claude"


def _resolve_args() -> list[str]:
    raw = os.getenv("CLAUDE_CODE_ACP_ARGS", "").strip()
    if not raw:
        return [
            "--print", "--output-format", "stream-json", "--verbose",
            "--dangerously-skip-permissions", "--no-session-persistence",
        ]
    return shlex.split(raw)


def _parse_claude_event(evt: dict) -> tuple[str, bool]:
    """Parse one NDJSON event from Claude Code CLI --output-format stream-json.

    Shapes::

        {"type":"assistant","message":{"content":[{"type":"text","text":"PONG"}]}}
        {"type":"result","result":"final text"}
        {"type":"system","subtype":"init",...}   (ignored)

    Returns ``(text, finished)``. A ``result`` event terminates the turn.
    """
    evt_type = evt.get("type")
    if evt_type == "assistant":
        message = evt.get("message") or {}
        texts: list[str] = []
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                txt = block.get("text", "")
                if txt:
                    texts.append(txt)
        return "".join(texts), False
    if evt_type == "result":
        result = evt.get("result", "")
        if isinstance(result, str) and result:
            return result, True
        return "", True
    if evt_type == "error":
        msg = evt.get("error") or evt.get("message") or "unknown error"
        raise RuntimeError(f"Claude Code CLI error: {msg}")
    return "", False


class _CCChatCompletions:
    def __init__(self, client: "ClaudeCodeClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _CCChatNamespace:
    def __init__(self, client: "ClaudeCodeClient"):
        self.completions = _CCChatCompletions(client)


class ClaudeCodeClient:
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
        self.api_key = api_key or "claude-code-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = command or _resolve_command()
        self._args = list(args or _resolve_args())
        self._cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = _CCChatNamespace(self)
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
        text = run_acp_subprocess(
            self._command,
            list(self._args),
            prompt,
            parse_event=_parse_claude_event,
            timeout_seconds=float(timeout or _DEFAULT_TIMEOUT_SECONDS),
            cwd=self._cwd,
            cli_name="Claude Code",
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
        return SimpleNamespace(choices=[choice], usage=usage, model=model or "claude-code-acp")


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = str(m.get("role") or "user").strip().lower()
        content = m.get("content") or ""
        if isinstance(content, list):
            texts = [
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content = "\n".join(texts)
        label = {"system": "System", "user": "User", "assistant": "Assistant"}.get(
            role, role.title()
        )
        if str(content).strip():
            parts.append(f"{label}:\n{content}")
    parts.append("Continue from the latest user message.")
    return "\n\n".join(parts)
