"""Upstream-contract drift canaries.

If any of these fail after a hermes update, the plugin's delegation pathway
may have changed: check what moved in the Hermes source before deploying a
pinned-route delegation again.
"""
import sys
from pathlib import Path

HERMES_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import HERMES_SRC  # noqa: E402
sys.path.insert(0, HERMES_SRC)


def test_lifecycle_request_accepts_reasoning_effort():
    """SubagentLaunchRequest must still carry reasoning_effort (the per-call
    reasoning override this plugin passes)."""
    from agent.subagent_lifecycle import SubagentLaunchRequest
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(SubagentLaunchRequest)}
    assert "reasoning_effort" in field_names, (
        "SubagentLaunchRequest no longer accepts reasoning_effort — the "
        "per-call reasoning pathway has changed upstream. Do NOT deploy this "
        "version of the plugin until retargeted."
    )


def test_lifecycle_request_maps_and_normalizes_effort():
    """The service must normalize effort via hermes' canonical parser and
    reject invalid values before a child is built."""
    from agent.subagent_lifecycle import SubagentLifecycleError
    from hermes_constants import parse_reasoning_effort
    assert parse_reasoning_effort("high") == {"enabled": True, "effort": "high"}
    assert parse_reasoning_effort(False) == {"enabled": False}
    assert parse_reasoning_effort("warp") is None  # unrecognized -> service rejects


def test_plugin_context_exposes_lifecycle_facade():
    """ctx.subagent_lifecycle is the plugin's launch surface; if the property
    moves, every delegate_task_pro call fails at handler time."""
    from hermes_cli.plugins import PluginContext
    assert hasattr(PluginContext, "subagent_lifecycle"), (
        "PluginContext.subagent_lifecycle is gone upstream — delegate_task_pro "
        "can no longer launch children. Retarget _lifecycle_service()."
    )


def test_stock_check_fn_importable():
    from tools.delegate_tool import check_delegate_requirements
    assert callable(check_delegate_requirements)


def test_inline_executor_table_importable():
    """The plugin hooks the agent loop via agent.inline_tool_executors.
    INLINE_TOOL_EXECUTORS — if the table moves, the tool loses parent_agent."""
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
    assert isinstance(INLINE_TOOL_EXECUTORS, dict)
    assert "delegate_task" in INLINE_TOOL_EXECUTORS


def test_sequential_path_consults_inline_table():
    """tool_executor sequential path checks INLINE_TOOL_EXECUTORS before the
    registry for non-delegate_task names."""
    src = (Path(HERMES_SRC) / "agent" / "tool_executor.py").read_text()
    assert "INLINE_TOOL_EXECUTORS" in src


def test_launch_never_touches_delegate_config_loader(plugin_module, monkeypatch):
    """Behavioral monkeypatch canary: a routed launch must not read or rebind
    tools.delegate_tool._load_config. The v1 plugin patched that loader to
    inject reasoning_effort (a core override that blocked catalog admission);
    the lifecycle path must make the patch structurally impossible — this test
    proves the old pathway stays dead, not just that a symbol is gone."""
    import tools.delegate_tool as dt

    class Boom(Exception):
        pass

    def _refuse(*_a, **_k):
        raise Boom("delegate_task_pro must not load delegation config directly")

    monkeypatch.setattr(dt, "_load_config", _refuse)

    class FakeLifecycle:
        def __init__(self):
            self.requests = []

        def launch(self, request):
            self.requests.append(request)
            return object()

        def wait(self, handle, timeout_seconds=None):
            return type("T", (), {"state": "SUCCEEDED", "timed_out": False})()

        def result(self, handle):
            return type("R", (), {"summary": "ok", "error_classification": None,
                                  "error_message": None})()

    class FakeCtx:
        def __init__(self, service):
            self.subagent_lifecycle = service

    svc = FakeLifecycle()
    monkeypatch.setattr(plugin_module, "_plugin_ctx", FakeCtx(svc))
    monkeypatch.setattr(plugin_module, "_load_plugin_config",
                        lambda: {"tiers": {"big": {"model": "m", "reasoning_effort": "high"}}})

    out = plugin_module._delegate_pro_tool({"goal": "g", "tier": "big"},
                                           parent_agent=object())
    assert len(svc.requests) == 1
    assert "Error" not in out
    assert svc.requests[0].reasoning_effort == "high"
    # _load_config still patched to refuse: proving the handler never called it.


def test_sequential_deadline_exempt_set_present(plugin_module):
    """The plugin exempts delegate_task_pro from the generic sequential tool
    deadline by extending agent.tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS
    at registration (mirroring the stock delegate_task exemption). If the module
    or set is renamed/removed upstream, the exemption silently stops working and
    long children are abandoned by the ~420s deadline with their completion
    undelivered — make that loud instead."""
    import agent.tool_executor as te

    if not hasattr(te, "_SEQUENTIAL_DEADLINE_EXEMPT_TOOLS"):
        raise AssertionError(
            "agent.tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS is gone upstream — "
            "delegation-pro's deadline exemption no-ops and children that outlive the "
            "sequential deadline are orphaned. Inspect the new exemption mechanism and "
            "retarget the registration hook in register()."
        )

    class FakeCtx:
        def register_tool(self, *a, **kw):
            pass

        def register_system_prompt_section(self, *a, **kw):
            pass

    before = frozenset(te._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS)
    prev_ctx = plugin_module._plugin_ctx
    try:
        plugin_module.register(FakeCtx())
    finally:
        plugin_module._plugin_ctx = prev_ctx  # never leak between tests
    after = te._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS
    assert "delegate_task_pro" in after, (
        "register() did not exempt delegate_task_pro from the sequential deadline; "
        "long children will be abandoned by the ~420s tool deadline."
    )
    # Additive: stock exemptions must survive the plugin's mutation.
    for name in ("delegate_task", "manage_connections"):
        if name in before:
            assert name in after, f"plugin registration dropped the stock exemption for {name}"


def test_plugin_context_exposes_system_prompt_section_api():
    from hermes_cli.plugins import PluginContext
    assert hasattr(PluginContext, "register_system_prompt_section")