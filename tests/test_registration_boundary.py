"""Registration boundary: load the plugin root __init__.py the way PluginManager
does and assert the declared tool set. No Hermes-plugin imports from the source
tree — this tests the INSTALLED-shape plugin, not bundled source.

Live registration with a real PluginManager is exercised by
`hermes plugins validate` / live dogfood; this file asserts the
loader boundary and the declared surface only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import _load_plugin_module, HERMES_SRC  # noqa: E402
if HERMES_SRC not in sys.path:
    sys.path.insert(0, HERMES_SRC)


def test_plugin_shape(plugin_module):
    assert plugin_module._TOOL_NAME == "delegate_task_pro"
    assert plugin_module.register  # callable registration entry point


def test_register_adds_tool_and_captures_ctx(plugin_module):
    registered = {}
    sections = {}

    class FakeCtx:
        def register_tool(self, name, toolset, schema, handler, check_fn=None, **kw):
            registered["name"] = name
            registered["toolset"] = toolset
            registered["schema"] = schema
            registered["handler"] = handler
            registered["check_fn"] = check_fn
            registered["description"] = kw.get("description", "")

        def register_system_prompt_section(self, id, content, **kw):
            sections["id"] = id
            sections["content"] = content

    ctx = FakeCtx()
    before = plugin_module._plugin_ctx
    try:
        plugin_module.register(ctx)
        # register() must capture the ctx so the handler can reach ctx.subagent_lifecycle.
        assert plugin_module._plugin_ctx is ctx
    finally:
        plugin_module._plugin_ctx = before  # never leak between tests

    assert registered["name"] == "delegate_task_pro"
    assert registered["toolset"] == "delegation"
    assert callable(registered["handler"])
    assert callable(registered["check_fn"])
    # The tool listing reads the top-level description; an empty one shows blank.
    assert registered["description"].strip(), "delegate_task_pro must register a non-empty description"
    assert "tier" in registered["description"]
    # System-prompt nudge steering agents to the pro tool.
    assert sections.get("id") == "delegation-pro"
    assert "delegate_task_pro" in (sections.get("content") or "")


def test_schema_surface(plugin_module):
    cfg = {"default_tier": "small", "tiers": {
        "small": {"model": "prov/mini", "when": "trivial"},
        "large": {"model": "prov/big", "when": "deep work"},
    }}
    schema = plugin_module._build_schema(cfg)
    # Function-calling shape: description at the schema top level (that is what
    # registry.get_definitions surfaces), parameters nested under "parameters".
    assert schema.get("description", "").strip(), "schema must carry a top-level description"
    assert "tier" in schema["description"]
    assert "parameters" in schema, "schema must nest the parameter object under 'parameters'"
    assert schema["parameters"]["type"] == "object"
    props = schema["parameters"]["properties"]
    # lifecycle-supported fields only
    for k in ("goal", "context", "tier", "model", "reasoning_effort"):
        assert k in props, k
    # goal is the one required parameter
    assert schema["parameters"]["required"] == ["goal"]
    # unsupported stock fields must NOT be advertised
    for k in ("tasks", "output_schema", "images", "action", "subagent_id",
              "message", "background"):
        assert k not in props, f"schema must not advertise unsupported '{k}'"
    # tier descriptions surface 'when' text and the default marker
    tier_desc = props["tier"]["description"]
    assert "trivial" in tier_desc and "deep work" in tier_desc
    assert "- small" in tier_desc and "(default)" in tier_desc


def test_schema_definition_shape_definitions_ready(plugin_module):
    """Guard the exact bug that made the tool render with no parameters: the
    registry builds {**schema, name}, so 'parameters' must be nested."""
    schema = plugin_module._build_schema({})
    fn = {**schema, "name": plugin_module._TOOL_NAME}
    assert "parameters" in fn
    assert isinstance(fn["parameters"].get("properties"), dict) and fn["parameters"]["properties"]


def test_schema_default_without_entry(plugin_module):
    cfg = {"default_tier": "medium", "tiers": {"small": {"model": "prov/mini"}}}
    desc = plugin_module._build_schema(cfg)["parameters"]["properties"]["tier"]["description"]
    assert "no tiers entry" in desc


def test_schema_advertises_explicit_disable_level(plugin_module):
    props = plugin_module._build_schema({})["parameters"]["properties"]
    assert "none" in props["reasoning_effort"]["enum"]