"""Upstream-contract drift canaries.

If any of these fail after a hermes update, the plugin's delegation pathway
may have changed: check what moved in the Hermes source before deploying a
pinned-route delegation again.
"""
import inspect
import sys
from pathlib import Path

HERMES_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import HERMES_SRC  # noqa: E402
sys.path.insert(0, HERMES_SRC)


def test_credentials_cfg_kwarg_still_exists():
    """delegate_task() must still accept credentials_cfg (the per-call route)."""
    from tools.delegate_tool import delegate_task
    sig = inspect.signature(delegate_task)
    assert "credentials_cfg" in sig.parameters, (
        "delegate_task no longer accepts credentials_cfg — the per-call routing "
        "pathway has changed upstream. delegation-pro needs a new hook; do NOT "
        "deploy this version of the plugin."
    )


def test_reasoning_effort_read_site_still_present():
    """_resolve_child_runtime must still read reasoning_effort from the
    delegation config section — the plugin's scoped patch targets this read."""
    src = (Path(HERMES_SRC) / "tools" / "delegate_tool_config.py").read_text()
    assert "reasoning_effort" in src, (
        "delegation reasoning_effort resolution no longer lives in "
        "delegate_tool_config.py — the scoped _load_config patch will no-op. "
        "Inspect where the read moved and retarget the patch."
    )
    assert 'delegation_cfg.get("reasoning_effort")' in src


def test_stock_check_fn_importable():
    from tools.delegate_tool import check_delegate_requirements
    assert callable(check_delegate_requirements)


def test_inline_executor_table_importable():
    """The plugin hooks the agent loop via agent.inline_tool_executors.
    INLINE_TOOL_EXECUTORS — if the table moves, the tool loses parent_agent."""
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
    assert isinstance(INLINE_TOOL_EXECUTORS, dict)
    assert "delegate_task" in INLINE_TOOL_EXECUTORS


def test_concurrent_path_consults_inline_table():
    """tool_executor sequential path checks INLINE_TOOL_EXECUTORS before the
    registry for non-delegate_task names."""
    src = (Path(HERMES_SRC) / "agent" / "tool_executor.py").read_text()
    assert "INLINE_TOOL_EXECUTORS" in src