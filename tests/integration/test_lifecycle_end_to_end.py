"""End-to-end integration: delegation-pro (lifecycle branch) on patched Hermes.

This test is opt-in because it needs a Hermes checkout whose
``SubagentLaunchRequest`` already carries ``reasoning_effort`` (PR #140727):
set ``HERMES_LIFECYCLE_E2E=1`` and ``HERMES_SRC`` to that checkout.

It runs the real plugin through the real PluginManager, the real inline tool
executor, the real SubagentLifecycleService and a real child AIAgent.

Two endpoints are supported:
  * default: a local OpenAI-compatible fake that records every request body and
    answers with a fixed completion (no network, no spend);
  * real: set ``HERMES_E2E_REAL_BASE_URL`` (and ``HERMES_E2E_REAL_MODEL``) to an
    OpenAI-compatible server. A local recording proxy forwards to it, so the
    test still asserts the exact outgoing effort while a real model answers.

What both prove:
  * the plugin launches through ctx.subagent_lifecycle (no _load_config patch),
  * the request carries the tier's reasoning_effort,
  * the child is constructed with the matching reasoning_config_override,
  * the outgoing provider request carries that effort (not the delegation default),
  * the child's reply comes back through the tool result.
"""
import http.client
import json
import os
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("HERMES_LIFECYCLE_E2E") != "1",
    reason="set HERMES_LIFECYCLE_E2E=1 (and HERMES_SRC=<checkout with PR #140727>)",
)

PLUGIN_DIR = Path(__file__).resolve().parents[2]

_REQUESTS: list[dict] = []
_LOCK = threading.Lock()


class _FakeOpenAI(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # silence
        pass

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            body = json.dumps({"object": "list", "data": [{"id": "fake-model", "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}
        with _LOCK:
            _REQUESTS.append(body)

        model = body.get("model", "fake-model")
        if body.get("stream"):
            chunks = [
                {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": int(time.time()),
                 "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
                {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": int(time.time()),
                 "model": model, "choices": [{"index": 0, "delta": {"content": "child-ok"}, "finish_reason": None}]},
                {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": int(time.time()),
                 "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            payload = b"".join(
                f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks
            ) + b"data: [DONE]\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        completion = {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "child-ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
        data = json.dumps(completion).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _proxy_handler_class(upstream_url: str):
    parts = urlsplit(upstream_url)

    class _RecordingProxy(BaseHTTPRequestHandler):
        """Record the request, then forward it to a real OpenAI-compatible server."""

        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # silence
            pass

        def _forward(self, method: str, body: bytes | None = None):
            headers = {
                k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "content-length", "connection")
            }
            conn = http.client.HTTPConnection(parts.hostname or "127.0.0.1", parts.port or 80, timeout=120)
            try:
                conn.request(method, self.path, body=body, headers=headers)
                response = conn.getresponse()
                payload = response.read()
                status = response.status
                response_headers = response.getheaders()
            finally:
                conn.close()

            self.send_response(status)
            for key, value in response_headers:
                if key.lower() not in ("transfer-encoding", "content-length", "connection"):
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self._forward("GET")

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw or b"{}")
            except Exception:
                body = {}
            with _LOCK:
                _REQUESTS.append(body)
            if not body.get("messages"):
                # Client connectivity probe: answer locally instead of shipping
                # an empty body to the real server.
                payload = b"{}"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self._forward("POST", raw)

    return _RecordingProxy


@pytest.fixture()
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture()
def real_proxy():
    upstream = os.environ.get("HERMES_E2E_REAL_BASE_URL")
    if not upstream:
        pytest.skip("set HERMES_E2E_REAL_BASE_URL to run against a real provider")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _proxy_handler_class(upstream))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _write_home(tmp_path: Path, base_url: str, model: str) -> Path:
    home = tmp_path / "hermes-home"
    plugins = home / "plugins"
    plugins.mkdir(parents=True)
    shutil.copytree(
        PLUGIN_DIR, plugins / "delegation-pro",
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "tests"),
    )
    (home / "config.yaml").write_text(
        f"""
plugins:
  enabled: [delegation-pro]
delegation:
  base_url: {base_url}
  api_key: fake-key
  model: {model}
  reasoning_effort: medium
delegation_pro:
  default_tier: small
  tiers:
    small:
      model: {model}
      reasoning_effort: none
      when: trivial work
    medium:
      model: {model}
      reasoning_effort: low
      when: default work
display:
  streaming: false
security:
  allow_private_urls: true
""",
        encoding="utf-8",
    )
    return home


def _run_cases(base_url: str, model: str, expected_content: str, goal: str, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(_write_home(tmp_path, base_url, model)))
    hermes_src = os.environ.get("HERMES_SRC")
    assert hermes_src, "HERMES_SRC must point at the Hermes checkout with PR #140727"
    if hermes_src not in sys.path:
        sys.path.insert(0, hermes_src)

    from agent.subagent_lifecycle import (
        SubagentLaunchRequest,
        SubagentLifecycleService,
        bind_subagent_parent,
    )
    assert "reasoning_effort" in SubagentLaunchRequest.__dataclass_fields__, (
        "HERMES_SRC does not contain the PR #140727 reasoning_effort field"
    )
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
    from hermes_cli.plugins import PluginManager
    from run_agent import AIAgent

    manager = PluginManager()
    manager.discover_and_load()
    try:
        executor = INLINE_TOOL_EXECUTORS.get("delegate_task_pro")
        assert executor is not None, "delegation-pro did not register delegate_task_pro"

        import tools.delegate_tool as delegate_tool
        load_config_before = delegate_tool._load_config
        launches: list[SubagentLaunchRequest] = []
        builds: list[dict] = []
        real_launch = SubagentLifecycleService.launch
        real_build = delegate_tool._build_child_agent

        def recording_launch(self, request, *args, **kwargs):
            if isinstance(request, SubagentLaunchRequest):
                launches.append(request)
            return real_launch(self, request, *args, **kwargs)

        def recording_build(**kwargs):
            builds.append(kwargs)
            return real_build(**kwargs)

        monkeypatch.setattr(SubagentLifecycleService, "launch", recording_launch)
        monkeypatch.setattr(delegate_tool, "_build_child_agent", recording_build)

        parent = AIAgent(
            base_url=base_url,
            api_key="fake-key",
            model=model,
            provider="custom",
            max_iterations=2,
            quiet_mode=True,
            platform="probe",
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
            disabled_toolsets=[],
        )
        parent.reasoning_config = {"enabled": True, "effort": "medium"}

        cases = [("small", "none"), ("medium", "low")]
        with bind_subagent_parent(parent):
            for tier, expected in cases:
                launches.clear()
                builds.clear()
                with _LOCK:
                    _REQUESTS.clear()

                result = executor(parent, {"goal": goal, "tier": tier}, None)

                assert isinstance(result, str), result
                assert expected_content in result, result
                assert launches, f"{tier}: plugin did not call SubagentLifecycleService.launch"
                assert launches[-1].reasoning_effort == expected, (
                    f"{tier}: launch request carried {launches[-1].reasoning_effort!r}, "
                    f"expected {expected!r}"
                )
                assert builds, f"{tier}: child was not built"
                from hermes_constants import parse_reasoning_effort
                assert builds[-1]["reasoning_config_override"] == parse_reasoning_effort(expected), (
                    f"{tier}: child reasoning_config_override was "
                    f"{builds[-1]['reasoning_config_override']!r}"
                )
                with _LOCK:
                    # Ignore the client's empty connectivity probe; only real
                    # completions carry messages.
                    bodies = [b for b in _REQUESTS if b.get("messages")]
                assert bodies, f"{tier}: child made no provider request"
                sent = [b.get("reasoning_effort") for b in bodies]
                assert expected in sent, (
                    f"{tier}: outgoing reasoning_effort {sent!r} never carried {expected!r} "
                    f"(delegation default medium leaked through?)"
                )
                print(
                    f"E2E {tier}: launch={launches[-1].reasoning_effort!r} "
                    f"override={builds[-1]['reasoning_config_override']!r} "
                    f"wire={sent!r} result={result.splitlines()[0][:80]!r}"
                )

        assert delegate_tool._load_config is load_config_before, (
            "tools.delegate_tool._load_config was replaced during the call"
        )
    finally:
        manager.unload()


def test_delegation_pro_launches_via_lifecycle_with_reasoning(fake_server, tmp_path, monkeypatch):
    _run_cases(fake_server, "fake-model", "child-ok", "Reply with exactly: child-ok", tmp_path, monkeypatch)


def test_delegation_pro_against_real_provider(real_proxy, tmp_path, monkeypatch):
    model = os.environ.get("HERMES_E2E_REAL_MODEL", "ling-3.0-tiny")
    _run_cases(real_proxy, model, "REAL_OK", "Reply with exactly: REAL_OK", tmp_path, monkeypatch)
