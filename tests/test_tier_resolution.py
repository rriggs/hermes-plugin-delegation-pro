"""Tier/route resolution matrix."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import SAMPLE_CFG  # noqa: E402


def test_known_tier(plugin_module):
    r = plugin_module._resolve_route({"tier": "large"}, SAMPLE_CFG)
    assert r["error"] is None
    assert r["credentials_cfg"] == {"model": "prov/big"}
    assert r["reasoning"] == {"enabled": True, "effort": "high"}


def test_tier_with_provider_and_base_url(plugin_module):
    r = plugin_module._resolve_route({"tier": "local"}, SAMPLE_CFG)
    assert r["error"] is None
    assert r["credentials_cfg"] == {"model": "bare-no-prefix", "base_url": "http://localhost:9999/v1"}


def test_unknown_tier_errors(plugin_module):
    r = plugin_module._resolve_route({"tier": "gigantic"}, SAMPLE_CFG)
    assert "Unknown tier 'gigantic'" in r["error"]
    assert r["credentials_cfg"] is None


def test_explicit_model_wins_over_tier(plugin_module):
    r = plugin_module._resolve_route({"tier": "large", "model": "other/special"}, SAMPLE_CFG)
    assert r["credentials_cfg"] == {"model": "other/special"}
    # tier-level reasoning does NOT leak when tier was not used for the route
    assert r["reasoning"] is None


def test_tier_without_model_is_not_pinned(plugin_module):
    cfg = {"tiers": {"x": {"when": "no model here"}}}
    r = plugin_module._resolve_route({"tier": "x"}, cfg)
    assert r["error"] is None
    assert r["credentials_cfg"] is None


def test_default_tier_applies_when_omitted(plugin_module):
    r = plugin_module._resolve_route({}, SAMPLE_CFG)
    assert r["error"] is None
    assert r["credentials_cfg"] == {"model": "prov/mid"}
    assert r.get("defaulted") is True


def test_no_default_tier_no_tier_stock_route(plugin_module):
    cfg = {"tiers": SAMPLE_CFG["tiers"]}
    r = plugin_module._resolve_route({}, cfg)
    assert r["error"] is None
    assert r["credentials_cfg"] is None


def test_explicit_effort_wins_over_tier(plugin_module):
    r = plugin_module._resolve_route({"tier": "large", "reasoning_effort": "low"}, SAMPLE_CFG)
    assert r["reasoning"] == {"enabled": True, "effort": "low"}


def test_effort_false_disables_thinking(plugin_module):
    r = plugin_module._resolve_route({}, {"tiers": {}})
    assert r["reasoning"] is None
    cfg = {"tiers": {"off": {"model": "m", "reasoning_effort": False}}}
    r = plugin_module._resolve_route({"tier": "off"}, cfg)
    assert r["reasoning"] == {"enabled": False}


def test_unknown_effort_string_is_not_validated_here(plugin_module):
    # parse via hermes_constants so unknown strings don't crash resolution;
    # strictness is hermes's job (clamp_effort tolerates unknown names).
    r = plugin_module._resolve_route({"reasoning_effort": "warp"}, SAMPLE_CFG)
    # hermes parse_reasoning_effort returns None for unrecognized bare strings
    assert r["reasoning"] is None


def test_malformed_config_fails_safe(plugin_module):
    r = plugin_module._resolve_route({"tier": "big"}, {"tiers": {"big": "not a dict"}})
    assert "Unknown tier" in r["error"]


# --- provider and model are SEPARATE keys (stock delegation-config shape) ---

def test_tier_provider_and_model_are_separate_keys(plugin_module):
    cfg = {"tiers": {"t": {"model": "deepseek/deepseek-v4.1-flash", "provider": "openrouter"}}}
    r = plugin_module._resolve_route({"tier": "t"}, cfg)
    # the slash in the model id is part of the id, NOT a provider prefix
    assert r["credentials_cfg"] == {"model": "deepseek/deepseek-v4.1-flash", "provider": "openrouter"}


def test_tier_without_provider_inherits_parent(plugin_module):
    cfg = {"tiers": {"t": {"model": "some-model"}}}
    r = plugin_module._resolve_route({"tier": "t"}, cfg)
    assert r["credentials_cfg"] == {"model": "some-model"}


def test_explicit_model_is_verbatim(plugin_module):
    r = plugin_module._resolve_route({"model": "deepseek/deepseek-v4.1-flash"}, {})
    assert r["credentials_cfg"] == {"model": "deepseek/deepseek-v4.1-flash"}


def test_tier_provider_base_url_and_model(plugin_module):
    cfg = {"tiers": {"t": {"model": "m", "provider": "hyper",
                           "base_url": "https://hyper.charm.land/v1"}}}
    r = plugin_module._resolve_route({"tier": "t"}, cfg)
    assert r["credentials_cfg"] == {"model": "m", "provider": "hyper",
                                    "base_url": "https://hyper.charm.land/v1"}