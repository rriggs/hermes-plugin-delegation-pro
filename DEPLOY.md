# Internal deployment (Mobilinkd fleet)

Our deployment does not wait for upstream. `integration` is the branch the fleet
runs: `main` plus the stock-wrapper work (0.3.0).

## What runs here

The plugin forwards every `delegate_task` argument untouched and adds `tier`,
`model` and `reasoning_effort`. Batching, background dispatch,
`action=list/steer/stop`, `output_schema` and `images` are the stock function's
own behavior, not a reimplementation.

The core knob it needs is a per-call `reasoning_effort` on `delegate_task`. Our
core branch adds it, so there is no monkeypatch and no narrowed surface. On a
stock core the call fails loudly instead of dropping the operator's effort.

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

A gateway restart is required: plugin code and the patched core modules are
imported at process start.

## Verify

```bash
HERMES_SRC=~/.hermes/hermes-agent PYTHONPATH=~/.hermes/hermes-agent \
  <pytest-with-hermes-deps> -q
```

43 tests: tier resolution, routing kwargs, schema parity against the live stock
contract, drift canaries on both core kwargs, registration boundary. The core
branch's own `tests/tools/test_delegate_launch_reasoning_kwarg.py` proves the
effort reaches the outgoing provider request.

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
