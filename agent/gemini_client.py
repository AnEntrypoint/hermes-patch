from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ACP_MARKER_BASE_URL = "acp://gemini"
_DEFAULT_TIMEOUT_SECONDS = 900.0


def _resolve_command() -> str:
    return os.getenv("GEMINI_ACP_COMMAND", "").strip() or "gemini"


def _resolve_args() -> list[str]:
    raw = os.getenv("GEMINI_ACP_ARGS", "").strip()
    return shlex.split(raw) if raw else ["--output-format", "stream-json", "--approval-mode", "yolo"]


def _extract_text(line: str) -> str:
    try:
        evt = json.loads(line)
    except Exception:
        return line.strip()
    evt_type = evt.get("type")
    if evt_type == "result":
        val = evt.get("response", "")
        if isinstance(val, str) and val.strip():
            return val.strip()
    if evt_type == "message":
        role = str(evt.get("role") or "").lower()
        if role not in ("", "assistant"):
            return ""
        content = evt.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") in ("text", None)
            ]
            joined = "\n".join(p for p in parts if p)
            if joined:
                return joined
    return ""


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
        text = self._run_prompt(prompt, model=model, timeout_seconds=float(timeout or _DEFAULT_TIMEOUT_SECONDS))
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

    def _run_prompt(self, prompt: str, *, model: str | None = None, timeout_seconds: float) -> str:
        cmd = [self._command] + self._args + ["--prompt", prompt]
        if model:
            cmd += ["--model", model]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=self._cwd,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start gemini command '{self._command}'. "
                "Install Gemini CLI or set GEMINI_ACP_COMMAND."
            ) from exc

        inbox: queue.Queue[str] = queue.Queue()

        def _reader() -> None:
            for line in proc.stdout:
                line = line.strip()
                if line:
                    inbox.put(line)

        threading.Thread(target=_reader, daemon=True).start()

        parts: list[str] = []
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if proc.poll() is not None and inbox.empty():
                break
            try:
                line = inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            text = _extract_text(line)
            if text:
                parts.append(text)

        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        if not parts:
            stderr_out = (proc.stderr.read() or "").strip()
            raise RuntimeError(f"gemini returned no content. stderr: {stderr_out[:500]}")
        return "\n".join(parts)


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
