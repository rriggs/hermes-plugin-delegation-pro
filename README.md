# hermes-plugin-delegation-pro

![Delegation Pro Logo](delegation_pro.jpeg)

Enhanced task delegation for Hermes Agent: per-call **model selection** (tiered)
and per-call **reasoning_effort**, without touching the Hermes source tree.

Upstream declined exposing these knobs on `delegate_task` itself
(NousResearch/hermes-agent#56891), so this ships as an external plugin that
piggybacks on the maintained internal plumbing (`credentials_cfg` per-call
routing — the same mechanism `/review` uses).

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
      model: "ollama-cloud/gemma4:31b"
      reasoning_effort: none
      when: "trivial lookups, file reads, greps, mechanical edits"
    medium:
      model: "ollama-cloud/minimax-m3"
      when: "default delegation: implementation, research, summaries"
    large:
      model: "ollama-cloud/deepseek-v4-pro:0813"
      reasoning_effort: high
      when: "reviews, planning, debugging loops, schema design"
```

- `model` — exact model the child runs on. Use LIVE model ids; if the provider
  retires a tag (404/410), delegation fails loudly rather than falling back
  (pinned routes intentionally don't borrow the parent's fallback chain).
  Prefer `provider:` + explicit ids over presets.
- Optional per-tier: `provider`, `base_url`, `reasoning_effort` (`none` disables
  thinking; any standard level pins it), `when` — surfaced in the tool schema so
  the agent picks tiers by description, not by hardcoding.
- `default_tier` — used when the tool call omits `tier`.

## Tool

`delegate_task_pro` mirrors `delegate_task`'s behavior (goal/context/tasks/
background/output_schema/images/action/steer/stop) and adds:

- `tier` — model tier from config (default: `default_tier`)
- `model` — explicit `provider/model` escape hatch, wins over tier
- `reasoning_effort` — per-call level, wins over the tier's level

The route is pinned per call via the internal `credentials_cfg` hook; children
are built inside the call, before the sync/background fork, so backgrounded
batches get the same route.

## Behavior and safety

- Fail-safe: a route resolution error (e.g. unknown tier) warns and proceeds
  with the stock route; the delegation is never lost.
- Upstream drift is caught by the plugin's `test_drift.py` canaries
  (`credentials_cfg` signature, reasoning-effort read site, inline-executor
  table). If a hermes update breaks one, don't trust the plugin until it's
  retargeted.
- The per-call reasoning override is a scoped patch of the delegation-config
  loader, restored after every call. `hermes plugins validate` will flag it
  under `no core override` — that lint targets catalog admission; for local
  installs it is advisory. This plugin is NOT intended for hub-catalog
  submission while the patch is in place.

## Known limits (v1)

- One route per call: a batch's tasks all run on the chosen tier. Mixed-tier
  batches mean separate calls.
- The parent-agent handoff uses the agent loop's inline-executor table — an
  additive registration, not an override.

MIT. See LICENSE.
