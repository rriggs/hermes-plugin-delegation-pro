"""Tier/route resolution matrix (lifecycle-launch shape)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import SAMPLE_CFG  # noqa: E402


def test_known_tier(plugin_module):
    r = plugin_module._resolve_route({"tier": "large"}, SAMPLE_CFG)
    assert r["warning"] is None
    assert r["model"] == "prov/big"
    assert r["provider"] == "prov"
    assert r["reasoning_effort"] == "high"


def test_tier_without_provider_inherits_parent(plugin_module):
    r = plugin_module._resolve_route({"tier": "small"}, SAMPLE_CFG)
    assert r["provider"] is None
    assert r["model"] == "prov/mini"


def test_unknown_tier_warns_not_errors(plugin_module):
    r = plugin_module._resolve_route({"tier": "gigantic"}, SAMPLE_CFG)
    assert "Unknown tier 'gigantic'" in r["warning"]
    assert r["model"] is None and r["provider"] is None
    assert r["reasoning_effort"] is None


def test_explicit_model_wins_over_tier(plugin_module):
    r = plugin_module._resolve_route({"tier": "large", "model": "other/special"}, SAMPLE_CFG)
    assert r["model"] == "other/special"
    # tier-level reasoning does NOT leak when tier was not used for the route
    assert r["reasoning_effort"] is None


def test_default_tier_applies_when_omitted(plugin_module):
    r = plugin_module._resolve_route({}, SAMPLE_CFG)
    assert r["warning"] is None
    assert r["model"] == "prov/mid"


def test_no_default_tier_no_tier_is_stock_route(plugin_module):
    cfg = {"tiers": SAMPLE_CFG["tiers"]}
    r = plugin_module._resolve_route({}, cfg)
    assert r["warning"] is None
    assert r["model"] is None and r["provider"] is None
    assert r["reasoning_effort"] is None


def test_explicit_effort_wins_over_tier(plugin_module):
    r = plugin_module._resolve_route({"tier": "large", "reasoning_effort": "low"}, SAMPLE_CFG)
    assert r["reasoning_effort"] == "low"


def test_explicit_effort_alone(plugin_module):
    r = plugin_module._resolve_route({"reasoning_effort": "low"}, {})
    assert r["reasoning_effort"] == "low"
    assert r["model"] is None


def test_effort_false_is_explicit_disable(plugin_module):
    # YAML False in tier config normalizes to the boolean False the lifecycle
    # request accepts as an explicit disable (never "false" the string).
    r = plugin_module._resolve_route({"tier": "off"}, SAMPLE_CFG)
    assert r["reasoning_effort"] is False


def test_effort_none_inherits(plugin_module):
    r = plugin_module._resolve_route({"tier": "medium"}, SAMPLE_CFG)
    assert r["reasoning_effort"] is None


def test_malformed_tier_entry_warns(plugin_module):
    r = plugin_module._resolve_route({"tier": "big"}, {"tiers": {"big": "not a dict"}})
    assert "Unknown tier" in r["warning"]


# --- provider and model are SEPARATE keys (stock delegation-config shape) ---

def test_tier_provider_and_model_are_separate_keys(plugin_module):
    cfg = {"tiers": {"t": {"model": "deepseek/deepseek-v4.1-flash", "provider": "openrouter"}}}
    r = plugin_module._resolve_route({"tier": "t"}, cfg)
    # the slash in the model id is part of the id, NOT a provider prefix
    assert r["model"] == "deepseek/deepseek-v4.1-flash"
    assert r["provider"] == "openrouter"


def test_explicit_model_is_verbatim(plugin_module):
    r = plugin_module._resolve_route({"model": "deepseek/deepseek-v4.1-flash"}, {})
    assert r["model"] == "deepseek/deepseek-v4.1-flash"
    assert r["provider"] is None