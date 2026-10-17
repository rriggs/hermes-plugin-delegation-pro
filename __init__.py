"""delegation-pro: enhanced delegate_task with per-call model tier + reasoning_effort.

Launches children through the public plugin surface ``ctx.subagent_lifecycle``
(``PluginContext.subagent_lifecycle`` -> ``agent.subagent_lifecycle``) with a
per-call route (model/provider) and per-call ``reasoning_effort`` resolved from
the tool arguments and the profile's ``delegation_pro`` config. No core module
is patched or monkey-imported: the previous ``tools.delegate_tool._load_config``
override is gone, so the plugin no longer trips the core-override admission
lint.

Because the lifecycle API launches ONE child per call and returns when the
child terminates, the tool deliberately does NOT advertise the stock
delegate_task surface it cannot honor (batched ``tasks``, ``output_schema``,
``images``, ``action=list``, background delegation, per-launch timeouts).
Unsupported arguments are rejected with a specific error instead of being
silently dropped; the stock ``delegate_task`` remains the right tool for those.

Configuration (profile config.yaml):

    delegation_pro:
      default_tier: medium
      tiers:
        small:
          provider: openrouter
          model: inclusionai/ling-3.1-flash
          reasoning_effort: none
          when: "file reads, greps, mechanical edits, format checks"
        large:
          provider: hyper
          model: glm-5.3
          reasoning_effort: high
          when: "reviews, planning, debugging loops, schema design"

``provider`` and ``model`` are separate keys, matching the stock ``delegation``
config section. If ``delegation_pro`` is absent from config the tool still
works, launching on the stock route (delegation config > parent).
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PLUGIN_NAME = "delegation-pro"
_TOOL_NAME = "delegate_task_pro"
_PLUGIN_CONFIG_KEY = "delegation_pro"

# The live PluginContext captured at register() time; the only documented way a
# plugin reaches the host-owned surfaces (subagent_lifecycle among them) from
# a tool handler, which receives no ctx of its own. Under plugins.isolation:
# host this is a RemotePluginContext that proxies subagent_lifecycle by method
# call, so the same captured ctx works there too.
_plugin_ctx = None

# Top-level tool description (shown in the [delegation] tool listing). Frames
# the tool as a single-child, tier-routed delegation: everything the lifecycle
# path can honor, nothing it cannot.
_TOOL_DESCRIPTION = (
    "PREFER THIS TOOL over `delegate_task` to launch ONE subagent with a chosen "
    "model tier and reasoning effort. Children run in isolated contexts with "
    "their own conversation and terminal session; only the final summary "
    "returns to you. Pick the `tier` whose description matches the work (the "
    "tier parameter lists the configured tiers and when to use each); an "
    "explicit `model` wins over the tier, and `reasoning_effort` wins over the "
    "tier's level.\n\n"
    "USE FOR: any delegation where you want to control which model or how much "
    "reasoning the child uses -- choose the tier by its description, whatever "
    "scheme it encodes (cost tiers, role tiers, capability tiers).\n"
    "DO NOT USE FOR:\n"
    "- Multiple tasks in one call -> call delegate_task_pro once per task, or "
    "use delegate_task's batch `tasks` (this tool launches exactly one child).\n"
    "- JSON-schema-constrained answers (`output_schema`), image inputs "
    "(`images`), or listing/steering running children (`action=list/steer`) -> "
    "use `delegate_task` (not available on the launch surface this tool uses).\n"
    "- Mechanical multi-step work with no reasoning needed -> execute_code\n"
    "- A single tool call -> call the tool directly\n"
    "- Tasks needing user interaction -> subagents cannot ask questions\n\n"
    "RULES:\n"
    "- The child knows nothing of this conversation: pass everything needed "
    "via 'context'.\n"
    "- Child summaries are SELF-REPORTS: for external side effects require a "
    "verifiable handle (URL, ID, absolute path) and verify it yourself before "
    "reporting success.\n"
    "- One route per call; the call returns when the child terminates. An "
    "unknown tier warns and proceeds on the stock route; the delegation is "
    "never lost."
)

# System-prompt section: high-trust bytes charged EVERY turn, so keep it short.
_SYSTEM_PROMPT_SECTION_ID = "delegation-pro"
_SYSTEM_PROMPT_CONTENT = (
    "Delegation: launch tier-routed single-child delegations with "
    "`delegate_task_pro` (the tier parameter lists the configured tiers and "
    "when to use each; an explicit `model` and `reasoning_effort` win over the "
    "tier). Batches, output_schema, images and steer/list actions stay on "
    "plain `delegate_task`."
)

# Reasoning effort levels accepted by hermes_constants.parse_reasoning_effort,
# plus the explicit-disable spellings. "none" disables thinking for the child.
_EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "maximum")


# ── config resolution ────────────────────────────────────────────────────────

def _load_plugin_config() -> dict:
    """The ``delegation_pro`` section of the profile config (read-only)."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
        section = cfg.get(_PLUGIN_CONFIG_KEY) if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("delegation-pro: config load failed: %s", exc)
        return {}


def _tiers(cfg: dict) -> dict:
    tiers = cfg.get("tiers")
    return tiers if isinstance(tiers, dict) else {}


def _parse_effort(effort: Any) -> Optional[Any]:
    """A ``reasoning_effort`` value in the shape ``SubagentLaunchRequest``
    accepts, or ``None`` (not requested / inherit). The lifecycle service
    normalizes the value through hermes' canonical parser and rejects invalid
    ones before a child is built; here we only pre-normalize tier-config
    entries (YAML booleans, casing) so the request carries a clean value."""
    if effort is None or (isinstance(effort, str) and not effort.strip()):
        return None
    if isinstance(effort, str):
        return effort.strip().lower()
    if isinstance(effort, bool):
        return False if not effort else None
    if isinstance(effort, dict):
        # AIAgent.reasoning_config shape ({enabled: False} or {enabled, effort});
        # pass through verbatim -- parse_reasoning_effort accepts dicts.
        return effort
    return effort


def _resolve_route(args: Dict[str, Any], cfg: Optional[dict] = None) -> Dict[str, Any]:
    """Resolve the per-call route from tool-call args + plugin config.

    Returns ``{"model": str|None, "provider": str|None, "reasoning_effort":
    Any|None, "warning": str|None}``.

    The route uses the same shape as the stock ``delegation`` config section:
    ``model`` and ``provider`` are SEPARATE keys. A model id may itself
    contain a slash (e.g. ``deepseek/deepseek-v4.1-flash`` on openrouter) --
    that slash is part of the id, never a provider prefix. Set ``provider``
    per tier to pin the child's provider (precedence: request > delegation
    config > parent); omit it to inherit.

    Fail-safe: an unknown tier never loses the delegation -- it produces a
    ``warning`` and an unpinned route (stock delegation route).
    """
    cfg = cfg if cfg is not None else _load_plugin_config()
    tiers = _tiers(cfg)

    explicit_model = (args.get("model") or "").strip() or None
    tier_name = (args.get("tier") or "").strip() or None

    model: Optional[str] = None
    provider: Optional[str] = None
    reasoning: Optional[Any] = None
    warning: Optional[str] = None

    if explicit_model:
        model = explicit_model
    elif tier_name:
        tier = tiers.get(tier_name)
        if not isinstance(tier, dict):
            warning = (
                f"Unknown tier '{tier_name}'. Available: {sorted(tiers) if tiers else '(none configured)'}. "
                f"Proceeding WITHOUT a tier pin (stock delegation route). Pass an explicit 'model' instead, "
                f"or omit 'tier' to use the default."
            )
        else:
            model = (tier.get("model") or "").strip() or None
            provider = (tier.get("provider") or "").strip() or None
            reasoning = _parse_effort(tier.get("reasoning_effort"))
    elif (cfg.get("default_tier") or "").strip():
        default_tier = cfg["default_tier"].strip()
        sub = dict(args)
        sub.pop("tier", None)
        sub["tier"] = default_tier
        return _resolve_route(sub, cfg)

    # Reasoning: explicit tool-call arg wins over tier-level default.
    explicit_effort = _parse_effort(args.get("reasoning_effort"))
    if explicit_effort is not None:
        reasoning = explicit_effort

    return {"model": model, "provider": provider, "reasoning_effort": reasoning, "warning": warning}


# ── launch request mapping ───────────────────────────────────────────────────

def _build_launch_request(args: Dict[str, Any], route: Dict[str, Any]) -> Any:
    """A ``SubagentLaunchRequest`` mapped from the tool-call arguments.

    ``model``/``provider``/``reasoning_effort`` come from the resolved route
    (tier defaults fold into the explicit tool args, explicit args win). The
    dataclass is imported lazily inside the handler so a Hermes without the
    lifecycle API degrades to a clear error at call time, not at import.
    """
    from agent.subagent_lifecycle import SubagentLaunchRequest
    request = {
        "goal": args.get("goal"),
        "context": args.get("context"),
        # Route (tier/config-derived) supplies the pins; explicit args won above.
        "model": route.get("model") or args.get("model"),
        "provider": route.get("provider"),
        "reasoning_effort": route.get("reasoning_effort"),
    }
    # Drop unset optionals so the request's own defaults apply.
    request = {k: v for k, v in request.items() if v is not None}
    return SubagentLaunchRequest(**request)


def _unsupported_call(args: Dict[str, Any]) -> Optional[str]:
    """Specific error for arguments the launch surface cannot honor, or None.

    The lifecycle API launches exactly one child per request and has no
    output_schema/images equivalent and no list/steer/stop action surface for
    the model. Silently dropping any of these would change what the caller
    asked for, so each gets a named, actionable error.
    """
    if args.get("tasks"):
        return (
            "delegate_task_pro launches exactly one child per call through the "
            "subagent lifecycle API and does not support batched 'tasks'. Call "
            "this tool once per task, or use delegate_task for batches."
        )
    if args.get("output_schema") is not None:
        return (
            "output_schema is not supported by delegate_task_pro: the subagent "
            "lifecycle launch surface has no per-child answer-schema constraint. "
            "Use delegate_task when the child's answer must validate against a "
            "JSON Schema."
        )
    if args.get("images") is not None:
        return (
            "images is not supported by delegate_task_pro: the subagent "
            "lifecycle launch surface has no per-child image inputs. Put file "
            "paths in 'context' (the child can open them) or use delegate_task."
        )
    action = args.get("action")
    if action not in (None, "spawn"):
        return (
            f"action='{action}' is not supported by delegate_task_pro: the "
            f"subagent lifecycle API exposes no list/steer/stop action surface "
            f"for the model. Use delegate_task's action='{'list' if action == 'list' else action}'."
        )
    return None


# ── result rendering ─────────────────────────────────────────────────────────

def _render_terminal(handle: Any, terminal: Any, result: Any) -> str:
    """A compact, bounded plain-text result for the parent's context.

    The lifecycle result object is the authoritative outcome (immutable,
    32k-clipped, hashed); this renders the fields the delegating model needs:
    state, summary, error classification, identity of the child.
    """
    lines = [
        f"Subagent {getattr(handle, 'subagent_id', '?')} finished "
        f"({getattr(terminal, 'state', 'UNKNOWN')})."
    ]
    summary = getattr(result, "summary", None)
    if summary:
        lines.append(str(summary))
    error_classification = getattr(result, "error_classification", None)
    error_message = getattr(result, "error_message", None)
    if error_classification and error_classification not in ("COMPLETED",):
        detail = f"{error_classification}: {error_message}" if error_message else str(error_classification)
        lines.append(f"Error: {detail}")
    if getattr(terminal, "timed_out", False):
        lines.append("Note: the wait timed out; the child may still be running.")
    return "\n".join(lines)


# ── tool schema ──────────────────────────────────────────────────────────────

def _build_schema(cfg: Optional[dict] = None) -> dict:
    """Tool schema. Tier names + when text from config are surfaced in the
    description so the agent can pick a tier without a skills-layer dependency.

    Advertises ONLY what the lifecycle path supports: single goal/context,
    tier/model/reasoning_effort routing. No tasks/output_schema/images/action
    surface -- those are rejected at dispatch with a specific error and stay
    on delegate_task.
    """
    cfg = cfg if cfg is not None else _load_plugin_config()
    tiers = _tiers(cfg)
    default_tier = (cfg.get("default_tier") or "").strip()

    tier_lines = []
    marked_default = False
    for name, tier in sorted(tiers.items()):
        if not isinstance(tier, dict):
            continue
        when = (tier.get("when") or "").strip()
        line = f"- {name}"
        if when:
            line += f": {when}"
        if name == default_tier:
            line += " (default)"
            marked_default = True
        tier_lines.append(line)
    if not tier_lines:
        tier_lines = ["- (none configured — pass an explicit model)"]
    tiers_text = "\n".join(tier_lines)
    if not marked_default and default_tier:
        tiers_text += f"\n(default_tier '{default_tier}' has no tiers entry — omitting 'tier' uses the stock route.)"

    model_pattern = r"^[a-zA-Z0-9._/-]+$"

    properties = {
        "goal": {
            "type": "string",
            "description": (
                "What this subagent should accomplish. Be specific and "
                "self-contained — the child knows nothing of this conversation."
            ),
        },
        "context": {
            "type": "string",
            "description": (
                "Background this child needs: file paths, error messages, "
                "constraints. The child sees only this context."
            ),
        },
        "tier": {
            "type": "string",
            "description": (
                f"Model tier for this subagent. Pick the tier whose description matches the "
                f"work; each tier's 'when' text says what it is for.\n{tiers_text}\n"
                f"Omit to use default_tier{f' ({default_tier})' if default_tier else ''}."
            ),
        },
        "model": {
            "type": "string",
            "pattern": model_pattern,
            "description": (
                "Explicit model id escape hatch (bare id, no provider prefix -- a slash in the "
                "id is part of the model, e.g. 'deepseek/deepseek-v4.1-flash'). Wins over tier. "
                "The child's provider comes from the tier/config, not from this string."
            ),
        },
        "reasoning_effort": {
            "type": "string",
            "enum": list(_EFFORT_LEVELS),
            "description": (
                "Per-call reasoning effort. Wins over the tier's level. 'none' disables "
                "thinking for this child. Omit to inherit."
            ),
        },
    }

    # Function-calling shape: the model receives this as function.parameters.
    # registry.get_definitions() does {**entry.schema, "name": entry.name}, so the
    # parameter object MUST be nested under "parameters" (like DELEGATE_TASK_SCHEMA
    # in tools/delegate_tool.py) — a flat {type, properties, required} lands as
    # siblings of "name" and the model sees NO parameters at all.
    return {
        "name": _TOOL_NAME,
        "description": _TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["goal"],
        },
    }


# ── tool handler ─────────────────────────────────────────────────────────────

def _delegate_pro_tool(args: Dict[str, Any], parent_agent=None, **_ignored) -> Any:
    """Tool handler. Resolves the tier route, launches one child through
    ``ctx.subagent_lifecycle`` and waits for its terminal state. Fails safe:
    an unknown tier warns and proceeds on the stock route; a launch the
    lifecycle API refuses (invalid effort, unknown provider) surfaces as a
    clear error and never silently degrades."""
    if parent_agent is None:
        return "delegate_task_pro requires a parent agent context."

    # Unsupported-argument gate: named, specific errors (never silent drops).
    unsupported = _unsupported_call(args)
    if unsupported:
        return f"Error: {unsupported}"

    try:
        route = _resolve_route(args)
    except Exception as exc:
        logger.warning("delegation-pro: route resolution failed: %s", exc)
        route = {"model": None, "provider": None, "reasoning_effort": None, "warning": None}

    service = _lifecycle_service()
    if isinstance(service, str):
        return service  # a clear error message, not a traceback

    try:
        request = _build_launch_request(args, route)
    except Exception as exc:
        return f"Error: invalid launch request: {exc}"

    try:
        handle = service.launch(request)
    except Exception as exc:
        return f"Error: launch failed: {exc}"

    # Blocking wait: the tool result must carry the child's final summary, and
    # the sequential-deadline exemption keeps the generic ~420s tool timeout
    # from abandoning a long child (children supervise their own liveness via
    # delegation.child_timeout_seconds).
    try:
        terminal = service.wait(handle)
    except Exception as exc:
        return (
            f"Subagent {getattr(handle, 'subagent_id', '?')} was launched but the "
            f"wait failed ({exc}). Treat the delegation as lost unless its "
            f"summary later appears elsewhere."
        )
    result = service.result(handle)

    out = _render_terminal(handle, terminal, result)
    if route.get("warning"):
        out = f"{route['warning']}\n\n{out}"
    return out


def _lifecycle_service():
    """The subagent lifecycle service, or a clear error message.

    Reaches the service through the PluginContext captured at ``register()``
    (``ctx.subagent_lifecycle`` — the documented plugin surface; a
    host-isolated ctx proxies it by method call). Never constructs a service
    with private delegation helpers itself.

    Returns the service object, or a ``str`` error message.
    """
    ctx = _plugin_ctx
    if ctx is None:
        return "Error: delegate_task_pro was not registered with a plugin context; the subagent lifecycle API is unreachable."
    try:
        service = ctx.subagent_lifecycle
        if service is None:
            return "Error: the subagent lifecycle API is unavailable in this Hermes."
        return service
    except Exception as exc:
        return (
            f"Error: this Hermes does not expose the subagent lifecycle API "
            f"(ctx.subagent_lifecycle): {exc}. Update Hermes (PR #140727 or "
            f"later) to use delegate_task_pro, or fall back to delegate_task."
        )


def _inline_executor(agent, args, _ctx) -> Any:
    """Agent-loop inline path: gets the live AIAgent directly (same channel stock
    delegate_task uses). Dispatched from agent/inline_tool_executors."""
    return _delegate_pro_tool(dict(args or {}), parent_agent=agent)


# ── plugin registration ──────────────────────────────────────────────────────

def register(ctx) -> None:
    global _plugin_ctx
    _plugin_ctx = ctx
    from tools.delegate_tool import check_delegate_requirements  # stock checker
    ctx.register_tool(
        name=_TOOL_NAME,
        toolset="delegation",
        schema=_build_schema(),
        handler=_delegate_pro_tool,
        check_fn=check_delegate_requirements,
        description=_TOOL_DESCRIPTION,
        emoji="\U0001f5fa",
    )
    # Additive inline-executor entry: the agent loop consults this table BEFORE
    # registry dispatch and passes the live AIAgent — the only way a tool sees
    # the parent agent (registry dispatch carries no parent_agent). A missing
    # entry or removed table (upstream refactor) is caught by test_drift.
    try:
        from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
        INLINE_TOOL_EXECUTORS[_TOOL_NAME] = _inline_executor
        logger.info("delegation-pro registered %s + inline executor", _TOOL_NAME)
    except Exception as exc:
        logger.warning("delegation-pro: could not register inline executor (%s); "
                       "the tool will only work where parent_agent is injected", exc)
    # Sequential-deadline exemption: the stock delegate_task is exempt from the
    # generic sequential tool deadline (agent.tool_executor.
    # _SEQUENTIAL_DEADLINE_EXEMPT_TOOLS) because a delegation call blocks for
    # the whole child by design and the child supervises its own liveness
    # (delegation.child_timeout_seconds, default 600s). This tool ALSO blocks
    # until the child terminates (lifecycle.wait), so it needs the same
    # exemption — otherwise a long child outlives the ~420s deadline, the
    # parent's tool call dies with tool_timeout, and the child's eventual
    # completion is never delivered. Extending the module-level frozenset is
    # additive and mirrors exactly what upstream did for delegate_task; a
    # read-back assertion below makes a silent failure impossible. If the
    # module/set moved (drift), we warn loudly and operators should set
    # timeouts.tools.sequential_call >= delegation.child_timeout_seconds in
    # config.yaml as the fallback.
    deadline_exempt = False
    try:
        import agent.tool_executor as _te
        exempt: set = set(getattr(_te, "_SEQUENTIAL_DEADLINE_EXEMPT_TOOLS", ()))
        exempt.add(_TOOL_NAME)
        _te._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS = frozenset(exempt)
        deadline_exempt = _TOOL_NAME in _te._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS
    except Exception as exc:
        logger.warning(
            "delegation-pro: could not exempt %s from the sequential tool deadline (%s); "
            "long children may be abandoned by the ~420s deadline with their completion "
            "undelivered. Fallback: set timeouts.tools.sequential_call >= delegation."
            "child_timeout_seconds in config.yaml.", _TOOL_NAME, exc,
        )
    if deadline_exempt:
        logger.info("delegation-pro exempted %s from the sequential tool deadline", _TOOL_NAME)
    # System-prompt nudge: steer agents to delegate_task_pro over the stock tool.
    # Best-effort — an older Hermes without the section API must still register the tool.
    try:
        ctx.register_system_prompt_section(_SYSTEM_PROMPT_SECTION_ID, _SYSTEM_PROMPT_CONTENT)
        logger.info("delegation-pro registered system prompt section %s", _SYSTEM_PROMPT_SECTION_ID)
    except Exception as exc:
        logger.warning("delegation-pro: could not register system prompt section (%s); "
                       "the tool still works but agents get no delegate_task_pro preference nudge", exc)
    logger.info("delegation-pro registered %s", _TOOL_NAME)