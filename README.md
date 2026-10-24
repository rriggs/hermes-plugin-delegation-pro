# hermes-plugin-delegation-pro

![Delegation Pro Logo](assets/delegation_pro.jpeg)

Enhanced task delegation for Hermes Agent: per-call **model selection** (tiered)
and per-call **reasoning_effort**, without patching the Hermes source tree.

Upstream declined exposing these knobs on the model-facing `delegate_task`
schema (NousResearch/hermes-agent#56891), so this ships as an external plugin
that wraps the stock function. Two plugin-facing core knobs carry the routing:
`credentials_cfg` (provider/model, the same mechanism `/review` uses) and
`reasoning_effort` (the same policy a launch-local `SubagentLaunchRequest`
applies). Everything else — batching, background dispatch, `action=list/steer/stop`,
`output_schema`, `images` — is the stock function's own behavior.

## Requirement

The core must accept `delegate_task(..., reasoning_effort=...)`. Stock Hermes
`main` does not; this line runs against the `internal/delegation-pro` core branch
(see `DEPLOY.md`). On a core without the kwarg the call fails loudly instead of
silently dropping the operator's effort.

## Install

```bash
hermes plugins install rriggs/hermes-plugin-delegation-pro
hermes plugins enable delegation-pro
```

Each profile needs its own install (sub-profiles do not inherit the default
profile's plugins).

## Configuration

Add a `delegation_pro` section to your profile's `config.yaml`:

```yaml
delegation_pro:
  default_tier: medium
  tiers:
    small:
      provider: openrouter
      model: inclusionai/ling-3.1-flash
      reasoning_effort: none
      when: "trivial lookups, file reads, greps, mechanical edits"
    medium:
      provider: openrouter
      model: deepseek/deepseek-v4.1-flash
      when: "default delegation: implementation, research, summaries"
    large:
      provider: hyper
      model: glm-5.3
      reasoning_effort: high
      when: "reviews, planning, debugging loops, schema design"
```

- `model` — exact model id the child runs on; `provider` — the provider that
  serves it. Set BOTH per tier, matching the stock `delegation` config section
  (`model:` + `provider:` as separate keys). A slash in the model id is part of
  the id (`deepseek/deepseek-v4.1-flash`), never a provider prefix. Omit
  `provider` (and `base_url`) to inherit the parent's provider. Use LIVE model
  ids; if the provider retires a tag (404/410), delegation fails loudly rather
  than falling back (pinned routes intentionally don't borrow the parent's
  fallback chain).
- Optional per-tier: `provider`, `base_url`, `reasoning_effort` (`none` disables
  thinking; any standard level pins it), `when` — surfaced in the tool schema so
  the agent picks tiers by description, not by hardcoding.
- `default_tier` — used when the tool call omits `tier`.

## Tool

`delegate_task_pro` mirrors `delegate_task`'s current behavior (the top-level
`tasks`/`action`/`subagent_id`/`message` surface, per-task
`goal`/`context`/`output_schema`/`images`, plus `group` when
`delegation.independent_completions` is on) and adds:

- `tier` — model tier from config (default: `default_tier`)
- `model` — explicit `provider/model` escape hatch, wins over tier
- `reasoning_effort` — per-call level, wins over the tier's level

The schema is derived from the core's own `delegate_task` parameters at
registration time and the three routing controls are added on top, so a core
change to the stock surface cannot leave the plugin advertising a stale copy
(`test_schema_parity.py` pins this).

The route is pinned per call via the internal `credentials_cfg` hook; children
are built inside the call, before the sync/background fork, so backgrounded
batches get the same route.

The plugin also registers a short system-prompt section steering the agent to
call `delegate_task_pro` instead of `delegate_task` — the stock description
cannot be edited (core override is gated), so the preference is stated from the
plugin side.

## Behavior and safety

- Fail-safe: a route resolution error (e.g. unknown tier) warns and proceeds
  with the stock route; the delegation is never lost.
- Upstream drift is caught by the plugin's `test_drift.py` canaries
  (`credentials_cfg` and `reasoning_effort` signatures, inline-executor table).
  If a hermes update breaks one, don't trust the plugin until it's retargeted.
- No core patch: the v1 monkeypatch of the delegation-config loader is gone. The
  reasoning policy travels as a function kwarg, so there is no rebind of
  `_load_config` and nothing to restore in a `finally`.
- `hermes plugins validate` still reports the sequential-deadline exemption and
  three private-core imports; those are catalog-admission lint, not runtime risk.

## Known limits

- One route per call: a batch's tasks all run on the chosen tier. Mixed-tier
  batches mean separate calls.
- The parent-agent handoff uses the agent loop's inline-executor table — an
  additive registration, not an override.
- The sequential-deadline exemption is still added by rebinding a core frozenset.
  It only matters for the synchronous paths (nested calls, no-result-consumer
  sessions); a top-level call in a gateway/TUI session returns the stock
  background handle immediately.

MIT. See LICENSE.
