import importlib.util, os, sys, types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
INIT_PATH = PLUGIN_DIR / "__init__.py"
HERMES_SRC = os.environ.get("HERMES_SRC", os.path.expanduser("~/.hermes/hermes-agent"))

def _load_plugin_module():
    if "hermes_plugins.delegation_pro" in sys.modules:
        return sys.modules["hermes_plugins.delegation_pro"]
    if HERMES_SRC not in sys.path:
        sys.path.insert(0, HERMES_SRC)
    if "hermes_plugins" not in sys.modules:
        ns_pkg = types.ModuleType("hermes_plugins")
        ns_pkg.__path__ = []
        sys.modules["hermes_plugins"] = ns_pkg
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.delegation_pro", str(INIT_PATH),
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.delegation_pro"
    mod.__path__ = [str(PLUGIN_DIR)]
    sys.modules["hermes_plugins.delegation_pro"] = mod
    spec.loader.exec_module(mod)
    return mod

@pytest.fixture(scope="session")
def plugin_module():
    return _load_plugin_module()

SAMPLE_CFG = {
    "default_tier": "medium",
    "tiers": {
        "small": {"model": "prov/mini", "reasoning_effort": "none",
                   "when": "trivial reads and greps"},
        "medium": {"model": "prov/mid", "when": "default work"},
        "large": {"model": "prov/big", "provider": "prov",
                   "reasoning_effort": "high",
                   "when": "reviews, planning, debugging"},
        "off": {"model": "m", "reasoning_effort": False},
    },
}