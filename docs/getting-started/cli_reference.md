# CLI Reference

The package provides two console entry points, both powered by [Hydra](https://hydra.cc).

## aao_demo

Run a task-runner demo.

```bash
aao_demo                                # default: pick_and_place
aao_demo --config-name cup_on_coaster   # any config in aao_configs/
aao_demo --list                         # list available configs
```

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
aao_demo --config-name stack_color_blocks rounds=3 env.batch_size=4 max_updates=500

# Override a nested key
aao_demo task.stages.0.param.pre_move.0.position="[0.4, 0.0, 0.1]"
```

### Output

Each run writes a `summary.json` to the Hydra output directory (`outputs/<date>/<time>/summary.json`) containing per-round success rates, completion steps, timing, and failure reasons.

## aao_eval

Run policy evaluation. Same Hydra config system as `aao_demo` but accepts an external policy.

```bash
aao_eval --config-name pick_and_place       # evaluate with ConfigDrivenDemoPolicy (default)
aao_eval --config-name policy_eval_mock     # mock backend evaluation
aao_eval --list                             # list available configs
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

When `policy` is omitted, `aao_eval` defaults to `auto_atom.ConfigDrivenDemoPolicy`, which replays the same primitive actions that `aao_demo` uses. See [Policy Evaluation](../tools/policy_evaluation.md) for the full API reference.

## Config resolution

Both entry points resolve Hydra configs from `./aao_configs/` relative to the current working directory. Run them from the project root.

The `--list` flag scans this directory and prints every available config name (one per `.yaml` file).
