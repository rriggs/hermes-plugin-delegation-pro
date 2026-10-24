"""Schema parity with the built-in delegate_task (issue #3, criterion 1).

``delegate_task_pro`` is a strict superset: every model-facing parameter of the
stock tool keeps its shape; the routing controls are additive. A core change that
adds or renames a stock parameter must fail this test rather than silently making
the plugin's schema stale.

Descriptions are excluded from the comparison: stock descriptions are rebuilt per
``get_definitions()`` call (dynamic overrides), and the plugin adds its tier text.
"""
import json
import sys
from typing import Any
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import HERMES_SRC, _load_plugin_module  # noqa: E402
plugin_module = _load_plugin_module()
if HERMES_SRC not in sys.path:
    sys.path.insert(0, HERMES_SRC)

from tools.delegate_tool import DELEGATE_TASK_SCHEMA  # noqa: E402

ROUTING_ONLY = {"tier", "model", "reasoning_effort"}


def _strip(node: Any) -> Any:
    """Shape only: drop every 'description' key recursively."""
    if isinstance(node, dict):
        return {k: _strip(v) for k, v in node.items() if k != "description"}
    if isinstance(node, list):
        return [_strip(v) for v in node]
    return node


def _stock_props():
    """The stock parameters as the core advertises them today (dynamic overrides
    included when available), so the comparison is against the live contract."""
    try:
        from tools.delegate_tool import _build_dynamic_schema_overrides
        dyn = (_build_dynamic_schema_overrides() or {}).get("parameters")
        if isinstance(dyn, dict) and isinstance(dyn.get("properties"), dict):
            return _strip(dyn)["properties"]
    except Exception:
        pass
    return _strip(DELEGATE_TASK_SCHEMA["parameters"])["properties"]


def _pro_props(monkeypatch):
    monkeypatch.setattr(plugin_module, "_load_plugin_config", lambda: {})
    return _strip(plugin_module._build_schema()["parameters"])["properties"]


def test_every_stock_parameter_survives(monkeypatch):
    missing = set(_stock_props()) - set(_pro_props(monkeypatch))
    assert not missing, f"stock parameters dropped from delegate_task_pro: {sorted(missing)}"


def test_additions_are_exactly_the_routing_controls(monkeypatch):
    added = set(_pro_props(monkeypatch)) - set(_stock_props())
    assert added == ROUTING_ONLY, f"schema additions: {sorted(added)}"


def test_required_set_matches_stock(monkeypatch):
    monkeypatch.setattr(plugin_module, "_load_plugin_config", lambda: {})
    pro = plugin_module._build_schema()["parameters"]
    assert sorted(pro.get("required", [])) == sorted(DELEGATE_TASK_SCHEMA["parameters"].get("required", []))


def test_stock_parameter_shapes_are_preserved(monkeypatch):
    stock = _stock_props()
    pro = _pro_props(monkeypatch)
    for name, spec in stock.items():
        assert json.dumps(pro[name], sort_keys=True) == json.dumps(spec, sort_keys=True), (
            f"parameter '{name}' changed shape"
        )
