"""Shared helpers for ACP subprocess providers (claude-code, kilocode, opencode, gemini).

These providers all spawn a local CLI via subprocess and stream back NDJSON
events. The cross-cutting concerns handled here:

- **Windows shim bypass.** npm installs CLIs as ``<name>.cmd`` batch shims that
  trampoline through cmd.exe. cmd.exe caps the invocation line at 8191 chars,
  which Hermes prompts routinely exceed. ``resolve_launch_command()`` detects
  the shim and invokes ``node.exe <entry>`` directly so the only remaining
  limit is the CreateProcess 32 KiB cap — and even that is avoided by piping
  the prompt via stdin instead of argv.

- **Stdin prompt piping.** Accepts prompts of any size on every platform.

- **Typed event extraction.** Each CLI emits its own NDJSON shape. The helper
  takes a callable that turns one parsed event into ``(text, finished)`` so
  the driver loop knows when to stop reading. Callers supply a
  provider-specific parser without re-implementing the launch/stream loop.

- **Deterministic stop.** As soon as a parser returns ``finished=True`` the
  child process is terminated and accumulated text is returned. This prevents
  upstream CLIs (kilo, opencode) from looping through tool turns when Hermes
  only wants one LLM completion.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional


EventParser = Callable[[dict], tuple[str, bool]]
"""Turn a parsed NDJSON event into ``(text_to_append, finished)``."""


def resolve_launch_command(command: str) -> tuple[str, list[str]]:
    """Bypass Windows npm .cmd shim by invoking node.exe directly.

    Returns ``(executable, prefix_args)``. On non-Windows, or when the command
    is not an npm shim we can see through, the first element is the original
    command and prefix_args is empty.
    """
    if not command or os.name != "nt":
        return command, []
    lowered = command.lower()
    if not (lowered.endswith(".cmd") or lowered.endswith(".bat")):
        return command, []

    shim_dir = os.path.dirname(command)
    # Typical npm shim resolves to: <shim_dir>/node_modules/<pkg>/bin/<bin-name>
    # Inspect the .cmd to recover the real entry script path.
    try:
        with open(command, "r", encoding="utf-8", errors="replace") as fh:
            shim_text = fh.read()
    except Exception:
        return command, []

    # The shim body looks like: `"%_prog%"  "%dp0%\node_modules\pkg\bin\name" %*`
    # Grep out the first path containing node_modules that resolves to a file
    # on disk. Accept any extension — gemini's entry is `bundle\gemini.js`,
    # kilo's is `bin\kilo`, claude's is `bin\claude`.
    node_flags: list[str] = []
    entry_rel = None
    for token in shim_text.split():
        unq = token.strip('"')
        # Capture --no-warnings=... style node flags the shim injects
        if unq.startswith("--") and not entry_rel:
            node_flags.append(unq)
            continue
        if "node_modules" in unq:
            candidate = unq.replace("%dp0%", shim_dir).replace("%~dp0", shim_dir)
            candidate = candidate.replace("%dp0%\\", shim_dir + os.sep)
            if os.path.isfile(candidate):
                entry_rel = candidate
                break
    if not entry_rel:
        return command, []

    node_exe = os.path.join(shim_dir, "node.exe")
    if not os.path.isfile(node_exe):
        node_exe = shutil.which("node") or "node"
    return node_exe, node_flags + [entry_rel]


def run_acp_subprocess(
    command: str,
    args: list[str],
    prompt: str,
    *,
    parse_event: EventParser,
    timeout_seconds: float,
    cwd: Optional[str] = None,
    cli_name: str = "ACP CLI",
) -> str:
    """Spawn ``command args``, pipe ``prompt`` via stdin, stream NDJSON back.

    Stops reading as soon as ``parse_event`` returns ``finished=True`` for any
    event, or when the child exits, or when the timeout elapses.

    Raises RuntimeError on spawn failure or empty output.
    """
    exe, prefix_args = resolve_launch_command(command)
    cmd = [exe] + prefix_args + args

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=cwd or os.getcwd(),
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Could not start {cli_name} command '{command}'. "
            "Install the CLI or set the appropriate *_ACP_COMMAND env var."
        ) from exc

    if proc.stdin is None or proc.stdout is None:
        try:
            proc.kill()
        except Exception:
            pass
        raise RuntimeError(f"{cli_name} process did not expose stdin/stdout pipes.")

    try:
        proc.stdin.write(prompt)
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass

    inbox: queue.Queue[dict] = queue.Queue()

    def _reader() -> None:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                inbox.put(json.loads(line))
            except Exception:
                continue

    threading.Thread(target=_reader, daemon=True).start()

    parts: list[str] = []
    finished = False
    deadline = time.time() + timeout_seconds
    while time.time() < deadline and not finished:
        if proc.poll() is not None and inbox.empty():
            break
        try:
            evt = inbox.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            text, done = parse_event(evt)
        except Exception:
            continue
        if text:
            parts.append(text)
        if done:
            finished = True

    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    if not parts:
        stderr_out = ""
        try:
            stderr_out = (proc.stderr.read() or "").strip()
        except Exception:
            pass
        raise RuntimeError(
            f"{cli_name} returned no content. stderr: {stderr_out[:500]}"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Shared event parsers for OpenCode-style stream-json (kilo + opencode share
# this shape since kilo is a fork of opencode).
# ---------------------------------------------------------------------------

def parse_opencode_style(evt: dict) -> tuple[str, bool]:
    """Parse one NDJSON event from kilo / opencode.

    Shape::

        {"type":"text","part":{"type":"text","text":"PONG"}}
        {"type":"step_finish","part":{"type":"step-finish","reason":"stop"}}
        {"type":"error","error":{...}}

    Returns ``(text, finished)``. The literal event-type string must not
    leak into output (a prior bug where _extract_text returned "text"
    instead of part.text on every chunk).
    """
    evt_type = evt.get("type")
    part = evt.get("part")

    if evt_type == "error":
        err = evt.get("error") or {}
        msg = err.get("data", {}).get("message") or err.get("message") or str(err)
        raise RuntimeError(f"CLI error: {msg}")

    if isinstance(part, dict):
        if part.get("type") == "step-finish":
            return "", part.get("reason") in ("stop", "end_turn", "tool-calls")
        if part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str) and text:
                return text, False

    if evt_type == "step_finish":
        return "", True

    return "", False


_INSTALLED_AGENT_SENTINEL: set[str] = set()


def ensure_hermes_llm_agent(config_dir: Path, *, tool_list: list[str] | None = None) -> None:
    """Write a zero-tool ``hermes-llm`` primary agent to ``<config_dir>/agents``.

    OpenCode-family CLIs (kilo, opencode) always invoke their default tool set
    unless an alternative agent is selected. Hermes calls them as a thin LLM
    provider, so we ship a minimal agent that disables every tool — the CLI
    then performs a single completion and exits with ``reason: stop``.

    Idempotent: the file is only written once per process per config_dir.
    """
    key = str(config_dir)
    if key in _INSTALLED_AGENT_SENTINEL:
        return
    tools = tool_list or [
        "bash", "read", "write", "edit", "list", "glob", "grep",
        "webfetch", "task", "todowrite", "todoread",
    ]
    tools_block = "\n".join(f"  {t}: false" for t in tools)
    body = (
        "---\n"
        "name: hermes-llm\n"
        "description: Stateless LLM completion proxy used by Hermes Agent. No tools.\n"
        "mode: primary\n"
        "tools:\n"
        f"{tools_block}\n"
        "---\n"
        "You are a pure LLM text-completion endpoint. Never call tools. "
        "Reply to the user's message directly with text and stop.\n"
    )
    try:
        agents_dir = config_dir / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        target = agents_dir / "hermes-llm.md"
        # Only overwrite if missing or our content has changed, so user edits
        # to other agents nearby aren't touched.
        if not target.exists() or target.read_text(encoding="utf-8") != body:
            target.write_text(body, encoding="utf-8")
        _INSTALLED_AGENT_SENTINEL.add(key)
    except Exception:
        # Non-fatal: caller may still succeed if user already installed the
        # agent manually, or if the CLI doesn't require it (gemini).
        pass
