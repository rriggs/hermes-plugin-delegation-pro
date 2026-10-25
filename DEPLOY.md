# Internal deployment (Mobilinkd fleet)

Our deployment does not wait for upstream. `integration` is the branch the fleet
runs: `main` plus the stock-wrapper work (0.3.0).

## What runs here

The plugin forwards every `delegate_task` argument untouched and adds `tier`,
`model` and `reasoning_effort`. Batching, background dispatch,
`action=list/steer/stop`, `output_schema` and `images` are the stock function's
own behavior, not a reimplementation.

The per-call `reasoning_effort` knob exists only on cores carrying our branch. When the
core lacks it, the plugin falls back to a scoped config patch: `tools.delegate_tool._load_config`
is wrapped once per process and merges the requested effort into the delegation config the
child builder reads, but ONLY for the calling thread (a `threading.local` gate), so
concurrent delegations and every child see the untouched config. On a core with the kwarg
the patch is never installed. If neither path is available the effort is dropped with an
`effort_warning` in the result instead of failing the call.

This fallback is a core override, which the plugin catalog forbids. It is accepted
deliberately: without it the effort knob is dead on every stock host, and passing the
kwarg there raises `TypeError` on every call (measured 2026-10-24 on alex, hai and
emerald, which took `delegate_task_pro` out entirely until the fallback landed).

Check before deploying to a new host:
`grep -n "def delegate_task(" -A 8 ~/.hermes/hermes-agent/tools/delegate_tool.py` and look
for `reasoning_effort` in the signature. Either answer works, but expect the log line
`installed a thread-scoped delegation.reasoning_effort patch` on stock cores.

## Core requirement

| Piece | Ref |
|---|---|
| Runtime checkout | `~/.hermes/hermes-agent` |
| Branch | `internal/delegation-pro` |
| Base | runtime `main` (carries 7 local deployment commits) |
| Adds | the five commits of NousResearch/hermes-agent#140727 plus the F1 fix (`rriggs/hermes-agent#1`), then the per-call `delegate_task(reasoning_effort=...)` commit |

Branch state is committed, not a working-tree patch, so
`git log main..internal/delegation-pro` is the authoritative diff.

Do not run `hermes update` while the checkout is on this branch: update works on
`main` and a branch switch is required first. Record the branch tip before any
update so the core patch can be re-applied.

## Install

The payload is `__init__.py`, `plugin.yaml`, `README.md`, `LICENSE`, `assets/`
from the `integration` branch. Per profile (plugins are not inherited):

```bash
git -C <repo> archive integration | tar -x -C /tmp/dpro
cp /tmp/dpro/{__init__.py,plugin.yaml,README.md,LICENSE} \
   ~/.hermes/profiles/<profile>/plugins/delegation-pro/
cp -r /tmp/dpro/assets ~/.hermes/profiles/<profile>/plugins/delegation-pro/
```

Then in that profile's `config.yaml`: `delegation-pro` under `plugins.enabled`
and a `delegation_pro:` tier block (`default_tier`,
`tiers.<name>.{provider,model,reasoning_effort,when}`).

### Deploy hazard: a duplicate plugin directory shadows the payload

Discovery builds one manifest per plugin key and lets a collision resolve to a single
directory (`resolve_manifest_winners`). A profile that carries both `delegation-pro/`
(the payload) and `hermes-plugin-delegation-pro/` (an older git checkout whose
`plugin.yaml` also says `name: delegation-pro`) serves the checkout, not the payload.
Measured 2026-10-24: the sven profile had been serving **0.1.1 from the git checkout**
the whole time. Copying 0.3.0 into `plugins/delegation-pro/` at 20:21:17 and restarting
the gateway at 20:27:39 changed nothing - the restart reloaded 0.1.1 from the same
directory, and `delegate_task_pro` behaved exactly like the pre-fix code: tier=local
blocked 45.45s and returned a completed child inline (stock `delegate_task` in the same
session returned `{status: dispatched, mode: background}` immediately), and an unknown
tier still emitted the old `Proceeding WITHOUT a tier pin` prefix instead of the JSON
`tier_warning` key.

Always check which directory won, right after install:

```bash
hermes -p <profile> plugins list | grep delegation-pro   # want: 0.3.0, source: user
grep "delegation-pro build" ~/.hermes/profiles/<profile>/logs/gateway.log | tail -1
```

The version alone is not enough: a `delegation-pro.bak-<ts>` left by an earlier deploy is
still discovered (only `_`/`.` prefixed dirs are skipped), carries the same
`name: delegation-pro`, sorts after the live directory, and wins. That is exactly what
happened on 2026-10-24: the backup held pre-fix 0.3.0, so `plugins list` reported 0.3.0
while the pre-fix code ran. Every deploy backup must be dot-prefixed:

```bash
mv ~/.hermes/profiles/<profile>/plugins/delegation-pro \
   ~/.hermes/profiles/<profile>/plugins/.delegation-pro.bak-$(date +%Y%m%dT%H%M%S)
```

The `delegation-pro build %r loaded from %s` line names the build and the exact file the
gateway loaded, which is the only unambiguous check.

A stale `hermes-plugin-delegation-pro/` (an old 0.1.1 git checkout) is the same hazard and
shows up as `0.1.1 / source: git`. Retire it the same way.

A gateway restart is still required to pick up new plugin code: `plugin_loader` returns
the module already in `sys.modules` (`_load_plugin_module_locked`) and `register()` runs
once per process, so a running gateway keeps serving the `__init__.py` it first imported.
Core-only patches (`tools/delegate_tool.py`) are re-imported per agent build and do land
without a restart.

## Verify

```bash
HERMES_SRC=~/.hermes/hermes-agent PYTHONPATH=~/.hermes/hermes-agent \
  <pytest-with-hermes-deps> -q
```

The runtime venv has the Hermes deps but no pytest; a standalone pytest can borrow
them: `PYTHONPATH=<runtime-venv>/lib/python3.14/site-packages ~/.local/bin/pytest -q`
(both are CPython 3.14). Pointing `HERMES_SRC` at an older install tree instead of
`~/.hermes/hermes-agent` fails the `reasoning_effort` drift canary — that canary is
doing its job.

46 tests: tier resolution, routing kwargs (including the background/sync dispatch
decision), schema parity against the live stock contract, drift canaries on both core
kwargs, registration boundary. The core branch's own
`tests/tools/test_delegate_launch_reasoning_kwarg.py` proves the effort reaches the
outgoing provider request.

Live check after restart: one `delegate_task_pro` call at a tier with
`reasoning_effort: none` and one at a tier with a level, then confirm the child
model and effort in the delegation live transcript.

## Rollback

Keep the previous plugin directory beside the new one
(`plugins/.delegation-pro.bak-<ts>`). Restore by moving it back and restarting the
gateway. Core rollback is `git switch main` in the runtime checkout plus restart.

## Issue #1 status (delivery after a dead parent call)

The incident's exact mechanism no longer applies to this line:

- the sequential tool deadline is disarmed for `delegate_task_pro` by the
  registration override (420.0 -> None once `register()` runs), and
- a top-level call in a gateway/TUI session now takes the stock background path,
  so it returns a dispatch handle instead of holding the turn open.

What still stands is the general guarantee the issue asks for: if the parent's
call dies by any other route (user interrupt, gateway restart, process crash, a
different budget), a still-running child's completion is orphaned - there is no
re-delivery and no sweep of completed-but-undelivered manifests. Issue #1's
second half (sweep + re-deliver) is not implemented in core.
