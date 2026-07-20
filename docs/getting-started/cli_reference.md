# CLI Reference

The package provides three console entry points. `aao-demo` and `aao-eval` are
powered by [Hydra](https://hydra.cc); `aao-info` introspects the task configs.

## aao-demo

Run a task-runner demo.

```bash
aao-demo                                # default: pick_and_place
aao-demo --config-name cup_on_coaster   # any config in aao_configs/
```

To discover which configs are runnable tasks, use [`aao-info`](#aao-info).

### Hydra overrides

| Override | Type | Default | Description |
|---|---|---|---|
| `rounds=N` | int | 1 | Number of demo rounds to run |
| `use_input=true` | bool | false | Pause between steps (press Enter to continue) |
| `max_updates=N` | int | 300 | Maximum update steps per round |
| `perf_count=true` | bool | false | Capture observations each step for performance analysis |
| `env.batch_size=N` | int | (from config) | Override the number of parallel environments |
| `task.seed=N` | int | (from config) | Override the randomization seed |
| `env.viewer.disable=true` | bool | false | Run headless (no viewer window) |

Any key present in the YAML config can be overridden on the command line following Hydra syntax:

```bash
# Multiple overrides
aao-demo --config-name stack_color_blocks rounds=3 env.batch_size=4 max_updates=500

# Override a nested key
aao-demo task.stages.0.param.pre_move.0.position="[0.4, 0.0, 0.1]"
```

### Output

Each run writes a `summary.json` to the Hydra output directory (`outputs/<date>/<time>/summary.json`) containing per-round success rates, completion steps, timing, and failure reasons.

## aao-eval

Run policy evaluation. Same Hydra config system as `aao-demo` but accepts an external policy.

```bash
aao-eval --config-name pick_and_place       # evaluate with ConfigDrivenDemoPolicy (default)
aao-eval --config-name policy_eval_mock     # mock backend evaluation
```

### Additional overrides

| Override | Type | Default | Description |
|---|---|---|---|
| `max_updates=N` | int | None | Maximum steps before stopping (None = unlimited) |
| `rounds=N` | int | 1 | Number of evaluation rounds |
| `use_input=true` | bool | false | Pause between steps |
| `get_obs=true` | bool | false | Call `capture_observation()` and pass to policy each step |

### Custom policy

Provide a `policy` section in the YAML config to use a custom policy:

```yaml
policy:
  _target_: my_package.MyPolicy
  checkpoint: /path/to/model.pt
```

When `policy` is omitted, `aao-eval` defaults to `auto_atom.ConfigDrivenDemoPolicy`, which replays the same primitive actions that `aao-demo` uses. See [Policy Evaluation](../tools/policy_evaluation.md) for the full API reference.

## aao-info

Introspect the **runnable tasks** in `aao_configs/`. Unlike a flat directory
listing, `aao-info` only reports configs that compose into a real task — i.e.
those with a non-empty `task.stages` after Hydra composition. Building-block
configs (bases, robot/eef definitions, mixins, variable files) are skipped
because they declare no stages.

For each task it reports the task name, the objects it manipulates, the
operations it performs, and a workflow generated from the ordered stages.

```bash
aao-info                    # list every runnable task
aao-info pick_and_place     # only the named config(s)
aao-info --json             # machine-readable output
aao-info --verbose          # also report configs skipped as non-tasks
```

| Argument | Description |
|---|---|
| `names...` | Optional config name(s) to inspect; default: all runnable tasks |
| `--json` | Emit a JSON array instead of the readable text report |
| `--config-dir DIR` | Config directory (default: `./aao_configs`) |
| `--verbose` | Print configs skipped as non-tasks or on composition errors (to stderr) |

Example output:

```
Runnable tasks (39):

press_three_buttons
  objects:    button_blue, button_green, button_pink
  operations: press
  workflow:
    1. press button_blue [press_blue]
    2. press button_green [press_green]
    3. press button_pink [press_pink]
```

Objects and operations are cross-checked: the report prefers the declared
`env.mask_objects` / `env.operations`, and adds a `note:` line when they differ
from the objects/operations actually referenced by the stages.

## Config resolution

`aao-demo` and `aao-eval` resolve Hydra configs from `./aao_configs/` relative to the current working directory, and `aao-info` scans the same directory. Run them from the project root.
