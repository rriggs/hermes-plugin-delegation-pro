# Internal deployment (Mobilinkd fleet)

Our deployment does not wait for upstream. `integration` is the branch the fleet
runs: `main` plus PR #2 (lifecycle launch, 0.2.0).

## Core requirement

`delegation-pro` 0.2.0 passes `reasoning_effort` on `SubagentLaunchRequest`. Stock
Hermes `main` does not have that field, so the tool refuses to launch there by
design (clear "update Hermes" error, never a silent degrade).

We run a patched core instead of waiting for upstream:

| Piece | Ref |
|---|---|
| Runtime checkout | `~/.hermes/hermes-agent` |
| Branch | `internal/delegation-pro` |
| Base | runtime `main` (carries 7 local deployment commits) |
| Adds | the five commits of NousResearch/hermes-agent#140727 plus the F1 fix (`rriggs/hermes-agent#1`) |

Those five commits add the field, carry it into child construction, and strip
conflicting raw `request_overrides` so an explicit launch effort actually wins on
the wire. Branch state is committed, not a working-tree patch, so
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
and a `delegation_pro:` tier block (`default_tier`, `tiers.<name>.{provider,model,reasoning_effort,when}`).

A gateway restart is required: plugin code and the patched core modules are
imported at process start.

## Verify

```bash
# unit + fake-endpoint E2E against the patched checkout
HERMES_SRC=~/.hermes/hermes-agent PYTHONPATH=~/.hermes/hermes-agent \
  <pytest-with-hermes-deps> tests/integration/test_lifecycle_end_to_end.py -q -s
```

Expected: the tier's effort (`none`, `low`, ...) appears in the outgoing provider
request captured by the fake endpoint, and the child's summary comes back through
the tool result.

Live check after restart: one `delegate_task_pro` call at a tier with
`reasoning_effort: none` and one at a tier with a level, then confirm the child
model and effort in the delegation live transcript.

## Rollback

Keep the previous plugin directory beside the new one
(`plugins/.delegation-pro.bak-<ts>`). Restore by moving it back and restarting the
gateway. Core rollback is `git switch main` in the runtime checkout plus restart.

## Known limits (open issue #3)

This line launches ONE child per call, waits for it, and rejects `tasks`,
`output_schema`, `images`, and `action=list/steer/stop` with explicit errors.
Stock `delegate_task` remains the path for those shapes until the parity work
lands. 0.1.1 (the monkeypatch line) still accepted them, so moving a profile to
0.2.0 removes batch/control-action support; deploy only to profiles that do not
depend on those.
