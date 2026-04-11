## [Unreleased]

### Fixed
- `gateway/status.py`: catch `OSError`/`SystemError` from `os.kill` on Windows when checking stale PID files (previously crashed gateway on startup after a dirty shutdown)
- `gateway/platforms/api_server.py`: allow same-origin browser requests without requiring `API_SERVER_CORS_ORIGINS` — the built-in UI at `GET /` can now call the API without a 403

### Added
- `opencode-acp` provider: subprocess client using `opencode run PROMPT --format json` (agent/opencode_client.py)
- `kilocode-acp` provider: subprocess client using `kilo run PROMPT --format json` (agent/kilocode_client.py)
- `gemini-acp` provider: subprocess client using `gemini --prompt PROMPT --output-format stream-json` (agent/gemini_client.py)
- All three registered in PROVIDER_REGISTRY (auth_type=acp), resolve_acp_provider_credentials, run_agent.py dispatch, and hermes model setup wizard
- Aliases: oc-acp, kilo-acp, kc-acp, gem-acp
- Web UI at `GET /`: webjsx + Ripple UI chat interface served by api_server (gateway/platforms/ui.html, no build step)
- Web UI markdown rendering: assistant messages now render via marked.js (GFM + line breaks) with styled lists, code blocks, blockquotes, and headings
- Web UI HTML responses: agent instructed via system message to reply in HTML fragments; UI renders directly via innerHTML, marked.js removed
