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


def test_stock_tasks_description_builder_present():
    """The plugin reuses stock's tasks-parameter text builder so its concurrency
    limit text does not drift."""
    from tools.delegate_tool import _build_tasks_param_description
    assert callable(_build_tasks_param_description)
    assert isinstance(_build_tasks_param_description(), str)


def test_stock_independent_completions_probe_present():
    """The plugin gates the per-task 'group' key on this stock probe."""
    from tools.delegate_tool_config import _get_independent_completions
    assert callable(_get_independent_completions)


def test_delegate_task_schema_uses_nested_parameters():
    """Pin the contract the plugin must match: DELEGATE_TASK_SCHEMA nests the
    parameter object under 'parameters' (not flat properties). If stock changes
    shape, the plugin's schema builders must follow."""
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA
    assert "parameters" in DELEGATE_TASK_SCHEMA
    assert "properties" in DELEGATE_TASK_SCHEMA["parameters"]


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
    plugin_module.register(FakeCtx())
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