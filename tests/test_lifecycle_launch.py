"""Launch behavior through a fake ctx.subagent_lifecycle: request mapping,
reasoning_effort normalization, unsupported-argument rejection, fail-safe.

These tests exercise the plugin's real handler against a fake lifecycle
service (behavior contract, not a grep canary): the fake records the
SubagentLaunchRequest the plugin builds and returns a scripted terminal state.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import _load_plugin_module, HERMES_SRC  # noqa: E402

plugin_module = _load_plugin_module()
if HERMES_SRC not in sys.path:
    sys.path.insert(0, HERMES_SRC)

from agent.subagent_lifecycle import (  # noqa: E402
    SubagentHandle,
    SubagentLifecycleError,
    SubagentState,
    SubagentTerminalState,
)
from hermes_constants import parse_reasoning_effort  # noqa: E402

import conftest  # noqa: E402


# ── fake lifecycle surface ───────────────────────────────────────────────────

def _make_handle(subagent_id="sa-1"):
    return SubagentHandle(
        contract_version=1, subagent_id=subagent_id, parent_session_id="s",
        correlation_id=None, created_at=123.0, provider=None, model=None,
        role="leaf", depth=1, capability="cap",
    )


class FakeLifecycle:
    """Records the launch request; scripts launch/wait/result outcomes."""

    def __init__(self, *, launch_error=None, terminal_state=SubagentState.SUCCEEDED,
                 result=None):
        self.requests = []
        self.launch_error = launch_error
        self.terminal_state = terminal_state
        self._result = result if result is not None else SimpleNamespace(
            handle=None, terminal_state=terminal_state, ready=True,
            summary="child done", error_classification=None, error_message=None,
        )

    def launch(self, request):
        if self.launch_error is not None:
            raise self.launch_error
        self.requests.append(request)
        return _make_handle()

    def wait(self, handle, timeout_seconds=None):
        return SubagentTerminalState(handle, self.terminal_state, True)

    def result(self, handle):
        return self._result


class FakeCtx:
    def __init__(self, service):
        self.subagent_lifecycle = service

    def register_tool(self, *a, **kw):
        pass

    def register_system_prompt_section(self, *a, **kw):
        pass


@pytest.fixture
def call(plugin_module, monkeypatch):
    """Handler invocation with a fake ctx + injected plugin config.

    Returns ``run(args, service=..., cfg=...) -> (output, service)``."""
    def _run(args, *, service=None, cfg=None):
        svc = service or FakeLifecycle()
        monkeypatch.setattr(plugin_module, "_plugin_ctx", FakeCtx(svc))
        monkeypatch.setattr(plugin_module, "_load_plugin_config",
                            (lambda: cfg) if cfg is not None else (lambda: {}))
        out = plugin_module._delegate_pro_tool(args, parent_agent=object())
        return out, svc
    return _run


# ── request mapping ──────────────────────────────────────────────────────────

def test_maps_goal_context_model_provider_effort_into_request(call):
    out, svc = call({"goal": "review the PR", "context": "repo at /x",
                     "model": "glm-5.3", "reasoning_effort": "high"})
    req = svc.requests[0]
    assert req.goal == "review the PR"
    assert req.context == "repo at /x"
    assert req.model == "glm-5.3"
    assert req.provider is None  # inherit: no tier/config provider
    # normalized per hermes_constants.parse_reasoning_effort
    assert parse_reasoning_effort(req.reasoning_effort) == {"enabled": True, "effort": "high"}
    assert "child done" in out
    assert "SUCCEEDED" in out


def test_tier_route_pins_model_provider_and_effort(call):
    out, svc = call({"goal": "g", "tier": "large"}, cfg=conftest.SAMPLE_CFG)
    req = svc.requests[0]
    assert req.model == "prov/big"
    assert req.provider == "prov"
    assert req.reasoning_effort == "high"


def test_default_tier_applies_when_tier_omitted(call):
    out, svc = call({"goal": "g"}, cfg=conftest.SAMPLE_CFG)
    req = svc.requests[0]
    assert req.model == "prov/mid"


def test_unrouted_launch_leaves_route_fields_default(call):
    out, svc = call({"goal": "g"})
    req = svc.requests[0]
    assert req.model is None and req.provider is None
    assert req.reasoning_effort is None
    assert "Error" not in out


# ── reasoning_effort semantics ───────────────────────────────────────────────

def test_explicit_disable_none_string(call):
    out, svc = call({"goal": "g", "reasoning_effort": "none"})
    req = svc.requests[0]
    assert parse_reasoning_effort(req.reasoning_effort) == {"enabled": False}


def test_explicit_disable_false_boolean(call):
    out, svc = call({"goal": "g", "reasoning_effort": False})
    req = svc.requests[0]
    assert req.reasoning_effort is False  # boolean disable passes through
    assert parse_reasoning_effort(req.reasoning_effort) == {"enabled": False}


def test_invalid_effort_surfaces_lifecycle_error(call):
    # "warp" is not a recognized level: the service rejects it before a child
    # is built; the handler surfaces that as a clear error, never a launch.
    svc = FakeLifecycle(launch_error=SubagentLifecycleError(
        "Invalid reasoning_effort: must be a recognized effort level or an explicit disable value"))
    out, svc = call({"goal": "g", "reasoning_effort": "warp"}, service=svc)
    assert svc.requests == []
    assert "Invalid reasoning_effort" in out
    assert "Error" in out


# ── unsupported-argument rejection ───────────────────────────────────────────

@pytest.mark.parametrize("args,needle", [
    ({"goal": "g", "tasks": [{"goal": "a"}, {"goal": "b"}]}, "exactly one child"),
    ({"goal": "g", "output_schema": {"type": "object"}}, "output_schema is not supported"),
    ({"goal": "g", "images": ["/tmp/x.png"]}, "images is not supported"),
    ({"goal": "g", "action": "list"}, "action='list' is not supported"),
    ({"goal": "g", "action": "steer", "subagent_id": "sa-1", "message": "hi"},
     "action='steer' is not supported"),
    ({"goal": "g", "action": "stop", "subagent_id": "sa-1"},
     "action='stop' is not supported"),
])
def test_unsupported_arguments_are_rejected_not_dropped(call, args, needle):
    out, svc = call(args)
    assert svc.requests == [], "no child may be launched for an unsupported shape"
    assert needle in out
    assert "delegate_task" in out  # each error points at the stock tool


def test_spawn_action_is_accepted(call):
    out, svc = call({"goal": "g", "action": "spawn"})
    assert len(svc.requests) == 1
    assert "Error" not in out


# ── fail-safe ────────────────────────────────────────────────────────────────

def test_unknown_tier_proceeds_unpinned_with_warning(call):
    out, svc = call({"goal": "g", "tier": "nope"},
                    cfg={"tiers": {"small": {"model": "prov/mini"}}})
    # delegation not lost: one launch happened, unpinned
    assert len(svc.requests) == 1
    req = svc.requests[0]
    assert req.model is None
    assert "Unknown tier 'nope'" in out
    assert "child done" in out


def test_launch_failure_is_a_clear_error(call):
    svc = FakeLifecycle(launch_error=SubagentLifecycleError(
        "No active Hermes parent session is available."))
    out, svc = call({"goal": "g"}, service=svc)
    assert svc.requests == []
    assert "Error: launch failed" in out
    assert "No active Hermes parent session" in out


def test_missing_ctx_is_a_clear_error(plugin_module, monkeypatch):
    monkeypatch.setattr(plugin_module, "_plugin_ctx", None)
    out = plugin_module._delegate_pro_tool({"goal": "g"}, parent_agent=object())
    assert "was not registered with a plugin context" in out


def test_handler_requires_parent_agent(plugin_module):
    out = plugin_module._delegate_pro_tool({"goal": "g"}, parent_agent=None)
    assert "parent agent context" in out


def test_failed_child_renders_error_fields(call):
    result = SimpleNamespace(
        handle=None, terminal_state=SubagentState.FAILED, ready=True,
        summary=None, error_classification="ERROR", error_message="child exploded",
    )
    svc = FakeLifecycle(terminal_state=SubagentState.FAILED, result=result)
    out, _ = call({"goal": "g"}, service=svc)
    assert "FAILED" in out
    assert "child exploded" in out


def test_interrupted_child_renders_state(call):
    out, _ = call({"goal": "g"},
                  service=FakeLifecycle(terminal_state=SubagentState.INTERRUPTED))
    assert "INTERRUPTED" in out