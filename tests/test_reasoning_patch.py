"""Scoped _load_config patch behavior."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import HERMES_SRC, _load_plugin_module  # noqa: E402
plugin_module = _load_plugin_module()
if HERMES_SRC not in sys.path:
    sys.path.insert(0, HERMES_SRC)
import tools.delegate_tool as dt  # noqa: E402


def test_patch_applies_and_restores(monkeypatch):
    original = dt._load_config
    sentinel = {"reasoning_effort": "untouched", "max_iterations": 3}
    monkeypatch.setattr(dt, "_load_config", lambda: dict(sentinel))

    seen = []
    with plugin_module._reasoning_effort_override({"enabled": True, "effort": "high"}):
        seen.append(dt._load_config())
    after = dt._load_config()

    assert seen == [{"reasoning_effort": {"enabled": True, "effort": "high"},
                     "max_iterations": 3}]
    # restored to the monkeypatched lambda, not the module's true original
    assert after == sentinel


def test_none_effort_is_noop(monkeypatch):
    original = dt._load_config
    with plugin_module._reasoning_effort_override(None):
        assert dt._load_config is original
    assert dt._load_config is original


def test_patch_restores_on_exception(monkeypatch):
    sentinel = {"reasoning_effort": "untouched"}
    patched_lambda = lambda: dict(sentinel)  # noqa: E731
    monkeypatch.setattr(dt, "_load_config", patched_lambda)
    try:
        with plugin_module._reasoning_effort_override({"enabled": True}):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # restored to the monkeypatched fn (the CM's `original`), not the true module fn
    assert dt._load_config is patched_lambda
    assert dt._load_config() == sentinel


def test_merged_config_preserves_other_keys(monkeypatch):
    monkeypatch.setattr(dt, "_load_config",
                        lambda: {"provider": "ollama-cloud", "model": "parent-model"})
    with plugin_module._reasoning_effort_override({"enabled": False}):
        merged = dt._load_config()
    assert merged["provider"] == "ollama-cloud"
    assert merged["model"] == "parent-model"
    assert merged["reasoning_effort"] == {"enabled": False}