"""Routing kwargs handed to the stock delegate_task — and the absence of core patches.

The plugin selects a route with ``credentials_cfg`` and a reasoning policy with the
core ``reasoning_effort`` kwarg. It must not rebind anything in Hermes core: the
whole stock surface (batch, background dispatch, control actions, output_schema,
images) is the stock function's own behavior.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import HERMES_SRC, _load_plugin_module  # noqa: E402
plugin_module = _load_plugin_module()
if HERMES_SRC not in sys.path:
    sys.path.insert(0, HERMES_SRC)
import tools.delegate_tool as dt  # noqa: E402


def _capture(monkeypatch):
    """Replace the stock call; return the kwargs it was handed."""
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return "STOCK-RESULT"

    monkeypatch.setattr(dt, "delegate_task", fake)
    return seen


def test_reasoning_and_credentials_are_passed_as_call_kwargs(monkeypatch):
    seen = _capture(monkeypatch)
    cfg = {"default_tier": "large", "tiers": {
        "large": {"provider": "prov", "model": "prov/big", "reasoning_effort": "high"}}}
    monkeypatch.setattr(plugin_module, "_load_plugin_config", lambda: cfg)

    out = plugin_module._delegate_pro_tool({"goal": "review the change"}, parent_agent=object())
    assert out == "STOCK-RESULT"
    assert seen["reasoning_effort"] == {"enabled": True, "effort": "high"}
    assert seen["credentials_cfg"]["model"] == "prov/big"
    assert seen["credentials_cfg"]["provider"] == "prov"


def test_explicit_disable_reaches_the_kwarg_as_a_disable(monkeypatch):
    seen = _capture(monkeypatch)
    plugin_module._delegate_pro_tool({"goal": "g", "reasoning_effort": "none"}, parent_agent=object())
    assert seen["reasoning_effort"] == {"enabled": False}


def test_no_reasoning_requested_leaves_the_kwarg_none(monkeypatch):
    seen = _capture(monkeypatch)
    monkeypatch.setattr(plugin_module, "_load_plugin_config", lambda: {})
    plugin_module._delegate_pro_tool({"goal": "g"}, parent_agent=object())
    assert seen["reasoning_effort"] is None


def test_stock_only_arguments_are_passed_through_untouched(monkeypatch):
    seen = _capture(monkeypatch)
    args = {
        "tasks": [{"goal": "a"}, {"goal": "b"}],
        "output_schema": {"type": "object"},
        "images": ["/tmp/x.png"],
        "action": "list",
        "subagent_id": "sa-1",
        "message": "hi",
    }
    plugin_module._delegate_pro_tool(args, parent_agent=object())
    for key, value in args.items():
        assert seen[key] == value, f"{key} must reach the stock tool unchanged"


def test_plugin_module_never_rebinds_the_config_loader():
    """No `_load_config` patch, imported or applied: the loader is the core one."""
    from tools.delegate_tool_config import _load_config as core_loader

    assert not hasattr(plugin_module, "_reasoning_effort_override")
    assert dt._load_config is core_loader


class _Agent:
    """Minimal parent with the one attribute the dispatch decision reads."""

    def __init__(self, depth=0):
        self._delegate_depth = depth


def test_top_level_delegation_dispatches_in_the_background(monkeypatch):
    """Stock rule (run_agent._dispatch_delegate_task): depth 0 -> background.

    Omitting `background` takes the stock function's Python default (False),
    which blocks the parent's turn for the whole batch and never hands back a
    dispatch handle.
    """
    seen = _capture(monkeypatch)
    plugin_module._delegate_pro_tool({"goal": "g"}, parent_agent=_Agent(depth=0))
    assert seen["background"] is True


def test_orchestrator_subagent_stays_synchronous(monkeypatch):
    """depth > 0 needs its workers' results inside its own turn, same as stock."""
    seen = _capture(monkeypatch)
    plugin_module._delegate_pro_tool({"goal": "g"}, parent_agent=_Agent(depth=1))
    assert seen["background"] is False


def test_unknown_tier_keeps_the_dispatch_handle_parseable(monkeypatch):
    """The fallback note must not prefix-corrupt the JSON dispatch envelope."""
    import json

    def fake(**kwargs):
        return json.dumps({"status": "dispatched", "mode": "background", "count": 1})

    monkeypatch.setattr(dt, "delegate_task", fake)
    monkeypatch.setattr(plugin_module, "_load_plugin_config", lambda: {"default_tier": "bogus", "tiers": {}})

    out = plugin_module._delegate_pro_tool({"goal": "g"}, parent_agent=_Agent())
    payload = json.loads(out)
    assert payload["status"] == "dispatched"
    assert "bogus" in payload["tier_warning"]
