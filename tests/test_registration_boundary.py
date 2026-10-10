"""Registration boundary: load the plugin root __init__.py the way PluginManager
does and assert the declared tool set. No Hermes-plugin imports from the source
tree — this tests the INSTALLED-shape plugin, not bundled source.

Live registration with a real PluginManager is exercised by
`hermes plugins validate` / live dogfood (Phase 5); this file asserts the
loader boundary and the declared surface only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import _load_plugin_module, HERMES_SRC  # noqa: E402
if HERMES_SRC not in sys.path:
    sys.path.insert(0, HERMES_SRC)
import tools.delegate_tool as dt  # noqa: E402


def test_plugin_shape(plugin_module):
    assert plugin_module._TOOL_NAME == "delegate_task_pro"
    assert plugin_module.register  # callable registration entry point


def test_register_adds_tool_to_fake_ctx(plugin_module):
    registered = {}

    class FakeCtx:
        def register_tool(self, name, toolset, schema, handler, check_fn=None, **kw):
            registered["name"] = name
            registered["toolset"] = toolset
            registered["schema"] = schema
            registered["handler"] = handler
            registered["check_fn"] = check_fn

    plugin_module.register(FakeCtx())
    assert registered["name"] == "delegate_task_pro"
    assert registered["toolset"] == "delegation"
    assert callable(registered["handler"])
    assert callable(registered["check_fn"])


def test_schema_surface(plugin_module):
    cfg = {"default_tier": "small", "tiers": {
        "small": {"model": "prov/mini", "when": "trivial"},
        "large": {"model": "prov/big", "when": "deep work"},
    }}
    schema = plugin_module._build_schema(cfg)
    assert schema["type"] == "object"
    props = schema["properties"]
    # stock-compatible passthrough fields
    for k in ("goal", "context", "tasks", "background", "output_schema", "images",
              "action", "subagent_id", "message"):
        assert k in props, k
    # pro fields
    for k in ("tier", "model", "reasoning_effort"):
        assert k in props, k
    # tier descriptions surface 'when' text and the default marker
    tier_desc = props["tier"]["description"]
    assert "trivial" in tier_desc and "deep work" in tier_desc
    assert "- small" in tier_desc and "(default)" in tier_desc


def test_schema_default_without_entry(plugin_module):
    cfg = {"default_tier": "medium", "tiers": {"small": {"model": "prov/mini"}}}
    desc = plugin_module._build_schema(cfg)["properties"]["tier"]["description"]
    assert "no tiers entry" in desc


def test_handler_requires_parent_agent(plugin_module):
    out = plugin_module._delegate_pro_tool({"tier": "large"}, parent_agent=None)
    assert "parent agent context" in out


def test_handler_passes_credentials_cfg(plugin_module, monkeypatch):
    """The handler must forward the resolved route to the stock delegate_task,
    with the reasoning patch active during the call, and restore afterwards."""
    cfg = {"default_tier": "", "tiers": {"big": {"model": "prov/huge", "reasoning_effort": "high"}}}
    monkeypatch.setattr(plugin_module, "_load_plugin_config", lambda: cfg)

    captured = {}

    def fake_stock(**kwargs):
        # capture what _load_config resolved to INSIDE the patch window
        captured.update(kwargs)
        captured["config_during_call"] = dt._load_config()
        return "OK"

    from conftest import HERMES_SRC
    if HERMES_SRC not in sys.path:
        sys.path.insert(0, HERMES_SRC)
    monkeypatch.setattr(dt, "delegate_task", fake_stock)

    class FakeParent:
        pass

    before = dt._load_config
    result = plugin_module._delegate_pro_tool({"tier": "big"}, parent_agent=FakeParent())
    assert result == "OK"
    assert captured["credentials_cfg"] == {"model": "prov/huge"}
    # reasoning patch active during the stock call
    assert captured["config_during_call"]["reasoning_effort"] == {"enabled": True, "effort": "high"}
    # and restored after
    assert dt._load_config is before


def test_handler_unknown_tier_proceeds_with_warning(plugin_module, monkeypatch):
    cfg = {"default_tier": "", "tiers": {}}
    monkeypatch.setattr(plugin_module, "_load_plugin_config", lambda: cfg)

    captured = {}
    import sys as _sys
    from conftest import HERMES_SRC
    if HERMES_SRC not in _sys.path:
        _sys.path.insert(0, HERMES_SRC)
    import tools.delegate_tool as dt
    monkeypatch.setattr(dt, "delegate_task", lambda **kw: captured.update(kw) or "OK")

    result = plugin_module._delegate_pro_tool({"tier": "nope"}, parent_agent=object())
    assert "Unknown tier" in result
    assert "OK" in result
    assert captured["credentials_cfg"] is None