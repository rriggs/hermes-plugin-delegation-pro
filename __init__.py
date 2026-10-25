"""delegation-pro: enhanced delegate_task with per-call model tier + reasoning_effort.

Wraps the stock ``tools.delegate_tool.delegate_task`` and passes a per-call route
via its ``credentials_cfg`` kwarg (the same mechanism /review uses) plus the
per-call ``reasoning_effort`` kwarg (the same policy a launch-local
``SubagentLaunchRequest.reasoning_effort`` applies). Nothing in Hermes core is
patched: the whole stock surface — batching, background dispatch, control
actions, ``output_schema``, ``images`` — stays exactly as the built-in tool.

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
config section. If ``delegation_pro`` is absent from config the tool still works,
falling back to the stock route for everything.
"""

import copy
import inspect
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PLUGIN_NAME = "delegation-pro"
_TOOL_NAME = "delegate_task_pro"
_PLUGIN_CONFIG_KEY = "delegation_pro"
# Build marker: logged at register() time with the resolved file path. A profile can
# carry several directories whose plugin.yaml claims the same name (a stale checkout, an
# un-hidden *.bak from a previous deploy); discovery loads exactly one of them, and
# without this line the only symptom is old behavior at the same version number.
_PLUGIN_BUILD = "2026-10-24-effort-guard"

# Top-level tool description (shown in the [delegation] tool listing). Mirrors
# the stock delegate_task framing so the model treats it as a drop-in, and adds
# the routing knobs this plugin exists for.
_TOOL_DESCRIPTION = (
    "PREFER THIS TOOL over `delegate_task` for delegation. It is a strict superset: "
    "the same parameters plus per-call model routing (`tier` or an explicit `model`) and "
    "`reasoning_effort`. With no `tier`/`model` it follows the profile's configured "
    "default tier. Use plain `delegate_task` only when you deliberately want no tier "
    "applied.\n\n"
    "Spawn subagents in isolated contexts; each gets its own conversation, "
    "terminal session, and toolset, and only its final summary returns to you. "
    "Pick the `tier` whose description matches the work (the tier parameter lists "
    "the configured tiers and when to use each).\n\n"
    "USE FOR: the same work as `delegate_task` (reasoning-heavy subtasks, work that "
    "would flood your context, independent parallel workstreams) plus any delegation "
    "where you want to control which model or how much reasoning the children use -- "
    "choose the tier by its description, whatever scheme it encodes (cost tiers, "
    "role tiers, capability tiers).\n"
    "DO NOT USE FOR (use these instead):\n"
    "- Mechanical multi-step work with no reasoning needed -> execute_code\n"
    "- A single tool call -> call the tool directly\n"
    "- Tasks needing user interaction -> subagents cannot ask questions\n\n"
    "RULES:\n"
    "- Children know nothing of this conversation: pass everything needed via 'context'.\n"
    "- Child summaries are SELF-REPORTS: for external side effects require a verifiable "
    "handle (URL, ID, absolute path) and verify it yourself before reporting success.\n"
    "- One route per call: every task in a batch runs on the chosen tier; mixed tiers "
    "mean separate calls.\n"
    "- An unknown tier warns and falls back to the stock route; the delegation is never lost."
)

# System-prompt section: high-trust bytes charged EVERY turn, so keep it short.
# This is the enforcement lever the tool description can only suggest (the stock
# delegate_task description cannot be edited: core override is gated).
_SYSTEM_PROMPT_SECTION_ID = "delegation-pro"
_SYSTEM_PROMPT_CONTENT = (
    "Delegation: delegate with `delegate_task_pro`, not `delegate_task`. It is a "
    "strict superset (same parameters) that also accepts `tier` (the tool's tier "
    "parameter lists the configured tiers and when to use each), an explicit "
    "`model`, and per-call `reasoning_effort`, and it falls back to the stock route "
    "when no tier/model is given. Choose the tier whose description matches the "
    "work. Call plain `delegate_task` only if `delegate_task_pro` is not available."
)

# Reasoning effort levels accepted by hermes_constants.parse_reasoning_effort.
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


def _parse_effort(effort: Any) -> Optional[Dict[str, Any]]:
    """Normalize an effort value to the dict that ``AIAgent.reasoning_config``
    expects. ``None`` = not requested / inherit parent. Uses the same parser as
    the stock delegation config path."""
    if effort is None or (isinstance(effort, str) and not effort.strip()):
        return None
    try:
        from hermes_constants import parse_reasoning_effort
        # parse_reasoning_effort(False) -> {"enabled": False} (disable thinking), keep that intact.
        return parse_reasoning_effort(effort)
    except Exception:
        return None


def _resolve_route(args: Dict[str, Any], cfg: Optional[dict] = None) -> Dict[str, Any]:
    """Resolve the per-call route from tool-call args + plugin config.

    Returns ``{"credentials_cfg": {…} or None, "reasoning": dict or None, "error": str or None}``.

    The route uses the same shape as the stock ``delegation`` config section:
    ``model`` and ``provider`` are SEPARATE keys. A model id may itself contain a
    slash (e.g. ``deepseek/deepseek-v4.1-flash`` on openrouter) -- that slash is
    part of the id, never a provider prefix. Set ``provider`` per tier to pin the
    child's provider; omit it (and ``base_url``) to inherit the parent's provider.
    """
    cfg = cfg if cfg is not None else _load_plugin_config()
    tiers = _tiers(cfg)

    explicit_model = (args.get("model") or "").strip() or None
    tier_name = (args.get("tier") or "").strip() or None

    route: Dict[str, Any] = {}
    reasoning = None
    error = None

    if explicit_model:
        route["model"] = explicit_model
    elif tier_name:
        tier = tiers.get(tier_name)
        if not isinstance(tier, dict):
            error = (
                f"Unknown tier '{tier_name}'. Available: {sorted(tiers) if tiers else '(none configured)'}. "
                f"Pass an explicit 'model' instead, or omit 'tier' to use the default."
            )
        else:
            model = (tier.get("model") or "").strip() or None
            if model:
                route["model"] = model
            provider = (tier.get("provider") or "").strip() or None
            if provider:
                route["provider"] = provider
            base_url = (tier.get("base_url") or "").strip() or None
            if base_url:
                route["base_url"] = base_url
            reasoning = _parse_effort(tier.get("reasoning_effort"))

    if not explicit_model and not tier_name:
        # No tier/model requested: use default_tier if configured, else stock.
        default_tier = (cfg.get("default_tier") or "").strip() or None
        if default_tier:
            sub = dict(args)
            sub.pop("tier", None)
            sub["tier"] = default_tier
            resolved = _resolve_route(sub, cfg)
            if resolved["error"] is None:
                resolved["defaulted"] = True
            return resolved

    # Reasoning: explicit tool-call arg wins over tier-level default.
    explicit_effort = _parse_effort(args.get("reasoning_effort"))
    if explicit_effort is not None:
        reasoning = explicit_effort

    credentials_cfg = dict(route) if route else None
    return {"credentials_cfg": credentials_cfg, "reasoning": reasoning, "error": error}


# ── tool handler ─────────────────────────────────────────────────────────────

def _stock_parameters() -> Dict[str, Any]:
    """The built-in ``delegate_task`` parameter object, derived — never copied.

    Uses the stock dynamic-override builder when it is available so the tasks text
    and the ``group`` gate match whatever the core currently advertises, and falls
    back to the static schema. Deriving the base is what keeps this plugin's schema
    a true superset instead of a snapshot that silently rots.
    """
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA

    base = copy.deepcopy(DELEGATE_TASK_SCHEMA["parameters"])
    try:
        from tools.delegate_tool import _build_dynamic_schema_overrides
        dynamic = (_build_dynamic_schema_overrides() or {}).get("parameters") or {}
        if isinstance(dynamic.get("properties"), dict):
            base = copy.deepcopy(dynamic)
    except Exception as exc:  # a moved builder must not break registration
        logger.warning("delegation-pro: falling back to the static delegate_task "
                       "schema (dynamic override builder unavailable: %s)", exc)
    return base


def _build_schema(cfg: Optional[dict] = None) -> dict:
    """Tool schema: the stock parameter object plus the routing controls.

    Tier names + ``when`` text from config are surfaced in the ``tier``
    description so the agent can pick a tier without a skills-layer dependency.
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

    params = _stock_parameters()
    properties = dict(params.get("properties") or {})
    properties["tier"] = {
        "type": "string",
        "description": (
            f"Model tier for this subagent. Pick the tier whose description matches the "
            f"work; each tier's 'when' text says what it is for.\n{tiers_text}\n"
            f"Omit to use default_tier{f' ({default_tier})' if default_tier else ''}."
        ),
    }
    properties["model"] = {
        "type": "string",
        "pattern": model_pattern,
        "description": (
            "Explicit model id escape hatch (bare id, no provider prefix -- a slash in the "
            "id is part of the model, e.g. 'deepseek/deepseek-v4.1-flash'). Wins over tier. "
            "The child's provider comes from the tier/config, not from this string."
        ),
    }
    properties["reasoning_effort"] = {
        "type": "string",
        "enum": list(_EFFORT_LEVELS),
        "description": "Per-call reasoning effort. Wins over the tier's level. Omit to inherit.",
    }
    params["properties"] = properties

    # Function-calling shape: the model receives this as function.parameters.
    # registry.get_definitions() does {**entry.schema, "name": entry.name}, so the
    # parameter object MUST be nested under "parameters" (like DELEGATE_TASK_SCHEMA
    # in tools/delegate_tool.py) — a flat {type, properties, required} lands as
    # siblings of "name" and the model sees NO parameters at all.
    #
    # Unlike stock delegate_task we cannot register dynamic_schema_overrides here:
    # PluginContext.register_tool() does not forward that kwarg (only raw
    # registry.register does), so the base is resolved once at registration time.
    return {
        "name": _TOOL_NAME,
        "description": _TOOL_DESCRIPTION,
        "parameters": params,
    }


def _stock_accepts(fn: Any, kwarg: str) -> bool:
    """True when ``fn`` can take ``kwarg``: declared, or a ``**kwargs`` sink.

    An unreadable signature returns True — the caller's TypeError retry is the
    backstop, and optimistically passing the kwarg keeps behaviour identical on
    cores that do carry it.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    if kwarg in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _attach_note(result: Any, key: str, note: str) -> Any:
    """Embed a plugin note in a stock tool result without breaking its shape.

    A background dispatch returns a JSON handle that the model and the runtime
    parse as JSON, so the note rides INSIDE the envelope under ``key``. A plain
    text result (or a non-dict payload) gets the note prefixed instead.
    """
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return f"{note}\n\n{result}"
    if isinstance(payload, dict):
        payload[key] = note
        return json.dumps(payload)
    return f"{note}\n\n{result}"


def _delegate_pro_tool(args: Dict[str, Any], parent_agent=None, **_ignored) -> Any:
    """Tool handler. Resolves the route and calls the stock ``delegate_task`` with
    ``credentials_cfg`` and, where the core supports it, ``reasoning_effort``. Every
    other argument is passed through untouched, so batching, background dispatch,
    control actions, ``output_schema`` and ``images`` behave exactly like the
    built-in tool. Fails safe: any plugin-side error falls back to the stock call."""
    if parent_agent is None:
        return "delegate_task_pro requires a parent agent context."

    from tools.delegate_tool import delegate_task as _stock

    try:
        route = _resolve_route(args)
    except Exception as exc:
        logger.warning("delegation-pro: route resolution failed: %s", exc)
        route = {"credentials_cfg": None, "reasoning": None, "error": None}

    # Unknown tier: don't lose the delegation. Warn in the result and proceed
    # unpinned (stock route), unless the caller can pick a valid tier by retrying.
    tier_warning = None
    if route.get("error"):
        tier_warning = route["error"]

    # The per-call reasoning_effort kwarg exists only on cores carrying the
    # delegation-pro deployment branch. A stock core has no such parameter, and
    # passing it raises TypeError on EVERY call -- so tier routing, the whole point
    # of the plugin, would be unavailable on those hosts. Probe the signature and
    # drop the knob there, with a note in the result; the TypeError retry below
    # covers a core whose signature we cannot read.
    effort = route.get("reasoning")
    effort_note = None
    if effort is not None and not _stock_accepts(_stock, "reasoning_effort"):
        effort_note = (
            "reasoning_effort was NOT applied: this Hermes core has no per-call "
            "reasoning_effort (that knob ships on the delegation-pro core branch). "
            "The tier's model/provider routing was still applied."
        )
        logger.warning("delegation-pro: core delegate_task() has no reasoning_effort "
                       "parameter; dropping the effort for this call")
        effort = None

    # Dispatch mode is the stock tool's own decision, so it must be supplied the
    # way the stock inline executor supplies it: run_agent._dispatch_delegate_task
    # passes ``background=not (depth > 0)``. Calling the stock function directly
    # without it takes the Python default (background=False), which silently turns
    # every top-level delegation into a blocking call: the parent's turn stalls for
    # the whole batch and the model never gets the dispatch handle. Same rule as
    # stock -- orchestrator subagents (depth > 0) stay synchronous because they
    # need their workers' results inside their own turn.
    kwargs = dict(
        goal=args.get("goal"),
        context=args.get("context"),
        tasks=args.get("tasks"),
        background=not (getattr(parent_agent, "_delegate_depth", 0) > 0),
        output_schema=args.get("output_schema"),
        images=args.get("images"),
        action=args.get("action"),
        subagent_id=args.get("subagent_id"),
        message=args.get("message"),
        parent_agent=parent_agent,
        credentials_cfg=route.get("credentials_cfg"),
    )
    if effort is not None:
        kwargs["reasoning_effort"] = effort

    try:
        result = _stock(**kwargs)
    except TypeError as exc:
        if "reasoning_effort" in str(exc) and "reasoning_effort" in kwargs:
            # Signature probe could not see it (decorator, **kwargs wrapper, a core
            # that moved the parameter): degrade instead of breaking delegation.
            logger.warning("delegation-pro: delegate_task() rejected reasoning_effort "
                           "despite the signature probe; retrying without it")
            kwargs.pop("reasoning_effort")
            effort_note = (
                "reasoning_effort was NOT applied: this Hermes core rejected the "
                "per-call kwarg. Tier model/provider routing was still applied."
            )
            result = _stock(**kwargs)
        else:
            logger.warning("delegation-pro: delegate_task() rejected the routing kwargs (%s)", exc)
            raise

    if tier_warning:
        note = f"{tier_warning}\n\nProceeding WITHOUT a tier pin (stock delegation route)."
        result = _attach_note(result, "tier_warning", note)
    if effort_note:
        result = _attach_note(result, "effort_warning", effort_note)
    return result


def _inline_executor(agent, args, _ctx) -> Any:
    """Agent-loop inline path: gets the live AIAgent directly (same channel stock
    delegate_task uses). Dispatched from agent/inline_tool_executors."""
    return _delegate_pro_tool(dict(args or {}), parent_agent=agent)


# ── plugin registration ──────────────────────────────────────────────────────

def register(ctx) -> None:
    from tools.delegate_tool import check_delegate_requirements  # stock checker
    logger.info("delegation-pro build %r loaded from %s", _PLUGIN_BUILD, __file__)
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
    # generic sequential tool deadline (agent/tool_executor.
    # _SEQUENTIAL_DEADLINE_EXEMPT_TOOLS) because a delegation call blocks for the
    # whole batch by design and its children supervise their own liveness. The
    # plugin tool dispatches through the same inline-executor path, so it must be
    # exempt too — otherwise a long child (child_timeout_seconds up to 600s)
    # outlives the ~420s deadline, the parent's tool call dies with tool_timeout,
    # and the child's eventual completion is never delivered (Milo, 2026-10-10:
    # child completed cleanly at 605s after the parent call was killed at 420s).
    # Extending the module-level frozenset is additive and mirrors exactly what
    # upstream did for delegate_task; a read-back assertion below makes a silent
    # failure impossible. If the module/set moved (drift), we warn loudly and
    # operators should set timeouts.tools.sequential_call >= delegation.
    # child_timeout_seconds in config.yaml as the fallback.
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