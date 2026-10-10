"""delegation-pro: enhanced delegate_task with per-call model tier + reasoning_effort.

Wraps the stock ``tools.delegate_tool.delegate_task`` and passes a per-call route
via its ``credentials_cfg`` kwarg (the same mechanism /review uses). Per-call
``reasoning_effort`` is applied by a scoped patch of the delegation-config loader
for the duration of a single delegate_task() call.

Configuration (profile config.yaml):

    delegation_pro:
      default_tier: medium
      tiers:
        small:
          model: "ollama-cloud/qwen3-8b"
          reasoning_effort: none
          when: "file reads, greps, mechanical edits, format checks"
        large:
          model: "nous/claude-sonnet-6"
          reasoning_effort: high
          when: "reviews, planning, debugging loops, schema design"

If ``delegation_pro`` is absent from config the tool still works, falling back
to the stock route for everything (an explicit ``model:`` is still honored).
"""

import contextlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PLUGIN_NAME = "delegation-pro"
_TOOL_NAME = "delegate_task_pro"
_PLUGIN_CONFIG_KEY = "delegation_pro"

# Top-level tool description (shown in the [delegation] tool listing). Mirrors
# the stock delegate_task framing so the model treats it as a drop-in, and adds
# the routing knobs this plugin exists for.
_TOOL_DESCRIPTION = (
    "Spawn subagents in isolated contexts; each gets its own conversation, "
    "terminal session, and toolset, and only its final summary returns to you. "
    "Same contract as `delegate_task`, plus per-call model routing: pick a `tier` "
    "(see the tier parameter for the configured tiers) or pass an explicit "
    "`model`, and set `reasoning_effort` per call.\n\n"
    "USE FOR: the same work as `delegate_task` (reasoning-heavy subtasks, work that "
    "would flood your context, independent parallel workstreams) when you also want "
    "to control which model or how much reasoning the children use -- cheap tiers for "
    "mechanical work, stronger tiers for reviews, planning, and debugging.\n"
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
    A pinned model with no provider prefix is passed through as-is; provider is
    optional in the config (model-only pins are valid — the child will use the
    parent's provider with this model, and the same-provider inheritance rules
    in _resolve_child_runtime apply).
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


# ── scoped reasoning_effort patch ─────────────────────────────────────────────

@contextlib.contextmanager
def _reasoning_effort_override(effort: Dict[str, Any]):
    """Patch the delegation-config loader so ``delegation.reasoning_effort``
    reads as ``effort`` for the duration of ONE delegate_task() call.

    _build_child_agent re-loads config itself and reads reasoning_effort from
    the global delegation section (tools/delegate_tool_config.py). The patch
    window is exact because children are constructed synchronously inside the
    delegate_task() call, before any background fork.
    """
    if effort is None:
        yield
        return

    import tools.delegate_tool as dt
    original = dt._load_config

    def patched() -> dict:
        base = original()
        merged = dict(base) if isinstance(base, dict) else {}
        merged["reasoning_effort"] = effort
        return merged

    dt._load_config = patched
    try:
        yield
    finally:
        dt._load_config = original


# ── tool handler ─────────────────────────────────────────────────────────────

def _build_schema(cfg: Optional[dict] = None) -> dict:
    """Tool schema. Tier names + when text from config are surfaced in the
    description so the agent can pick a tier without a skills-layer dependency."""
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

    return {
        "type": "object",
        "description": _TOOL_DESCRIPTION,
        "properties": {
            "goal": {"type": "string", "description": "What this subagent should accomplish. Be specific and self-contained."},
            "context": {"type": "string", "description": "Background this child needs: file paths, error messages, constraints."},
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string"},
                        "context": {"type": "string"},
                    },
                    "required": ["goal"],
                    "additionalProperties": False,
                },
            },
            "tier": {
                "type": "string",
                "description": (
                    f"Model tier for this subagent. Choose the cheapest tier that can do the "
                    f"acceptance criteria; the description of each tier says when to use it.\n{tiers_text}\n"
                    f"Omit to use default_tier{f' ({default_tier})' if default_tier else ''}."
                ),
            },
            "model": {
                "type": "string",
                "pattern": model_pattern,
                "description": (
                    "Explicit model escape hatch, 'provider/model' or bare model id. Wins over tier. "
                    "Default: inherit the tier route or the parent's model."
                ),
            },
            "reasoning_effort": {
                "type": "string",
                "enum": list(_EFFORT_LEVELS),
                "description": "Per-call reasoning effort. Wins over the tier's level. Omit to inherit.",
            },
            "background": {
                "type": "boolean",
                "description": "Run in background; results re-enter as a new message when subagents finish.",
            },
            "output_schema": {"type": "object", "description": "Optional JSON Schema the child's final answer must validate against."},
            "images": {"type": "array", "items": {"type": "string"}, "description": "Image paths/URLs the child must SEE."},
            "action": {"type": "string", "enum": ["spawn", "list", "steer", "stop"]},
            "subagent_id": {"type": "string", "description": "Target for action='steer'/'stop'."},
            "message": {"type": "string", "description": "For action='steer': the course correction."},
        },
        "required": [],
    }


def _delegate_pro_tool(args: Dict[str, Any], parent_agent=None, **_ignored) -> Any:
    """Tool handler. Resolves the route, applies the reasoning patch window, and
    calls the stock delegate_task with credentials_cfg. Fails safe: any
    plugin-side error falls back to the stock call (no overrides)."""
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

    kwargs = dict(
        goal=args.get("goal"),
        context=args.get("context"),
        tasks=args.get("tasks"),
        background=args.get("background"),
        output_schema=args.get("output_schema"),
        images=args.get("images"),
        action=args.get("action"),
        subagent_id=args.get("subagent_id"),
        message=args.get("message"),
        parent_agent=parent_agent,
        credentials_cfg=route.get("credentials_cfg"),
    )

    try:
        with _reasoning_effort_override(route.get("reasoning")):
            result = _stock(**kwargs)
    except TypeError:
        # credentials_cfg kwarg removed upstream — fail safe: stock call without it.
        logger.warning("delegation-pro: delegate_task() did not accept credentials_cfg; "
                       "calling stock without overrides (drift!)")
        kwargs.pop("credentials_cfg", None)
        raise  # let the stock error surface, do not silently drop a route the caller asked for

    if tier_warning:
        result = f"{tier_warning}\n\nProceeding WITHOUT a tier pin (stock delegation route).\n\n{result}"
    return result


def _inline_executor(agent, args, _ctx) -> Any:
    """Agent-loop inline path: gets the live AIAgent directly (same channel stock
    delegate_task uses). Dispatched from agent/inline_tool_executors."""
    return _delegate_pro_tool(dict(args or {}), parent_agent=agent)


# ── plugin registration ──────────────────────────────────────────────────────

def register(ctx) -> None:
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
    logger.info("delegation-pro registered %s", _TOOL_NAME)