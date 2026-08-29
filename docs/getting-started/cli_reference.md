# CLI Reference

The package provides four console entry points. `aao-demo` and `aao-eval` are
powered by [Hydra](https://hydra.cc), `aao-unidoor-sweep` orchestrates bounded
Hydra multiruns, and `aao-info` introspects the task configs.

## `scripts/run_tests_safe.py`

Use the repository test runner when running more than a small focused test. It
executes test files in isolated, resource-bounded subprocesses. Up to
`--max-concurrency` batches may run at once (default: `4`) when the host can
support them; each batch remains independently bounded, so a simulator crash,
thread leak, or memory-heavy test cannot take down the whole pytest invocation.
The requested value is an upper bound, not a promise: the runner records and
uses a lower effective value when the selected CPU set, host/cgroup available
memory, `batch-mode`, launcher guarantees, or shared CUDA devices require a
clamp. The
default CPU set selects one available core, so use a multi-core set such as
`--cpu-set=0-3` (when those CPUs are available) when allowing four CPU-bound
batches is appropriate. The default launcher is a user `systemd` scope with a
CPU quota, cgroup memory/task limits, no swap, and a per-batch wall-clock limit.
If a user systemd manager is unavailable, `auto` uses the explicitly weaker
`prlimit` fallback and reduces concurrency to one batch.
With the default 6 GiB per-batch memory ceiling, a machine with limited free RAM
may still be clamped to one; use `--dry-run` to inspect the effective value before
starting a larger run.

The script never enables `pytest-xdist` or accepts `-n`/`--dist`; bounded
parallelism is at the batch level, while each batch itself runs one ordinary
pytest process. It neutralizes repository-level pytest `addopts` so a hidden
parallel setting cannot bypass the runner; pass any desired safe options
explicitly through `--pytest-args` or `PYTEST_ADDOPTS`. The latter are copied
into the recorded command, while parallel, `addopts`-override, and caller-owned
JUnit options are rejected. All repository-level `addopts` entries are
intentionally ignored; copy any non-default options you need into
`--pytest-args`.

Run it from the repository root with the project interpreter:

```bash
PYTHON=/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python

# All pytest-discoverable files, one bounded batch per file (the default).
# Up to four batches may overlap when the resource plan permits it.
$PYTHON scripts/run_tests_safe.py

# A focused run with up to four bounded batches; quote a space-separated list
# or use commas. The CPU set makes four slots available when resources allow.
$PYTHON scripts/run_tests_safe.py \
  --test-targets='tests/test_stage_execution.py tests/test_demo_eval_parity.py' \
  --batch-size=1 \
  --max-concurrency=4 \
  --cpu-set=0-3

# Force strict serial execution for simulator/GPU-sensitive tests
$PYTHON scripts/run_tests_safe.py \
  --test-targets='tests/test_stage_execution.py tests/test_demo_eval_parity.py' \
  --max-concurrency=1

# Inspect resolved targets and commands without starting pytest
$PYTHON scripts/run_tests_safe.py \
  --test-targets='tests/test_stage_execution.py tests/test_demo_eval_parity.py' \
  --max-concurrency=4 \
  --cpu-set=0-3 \
  --dry-run

# Stop after the first failed/timeout batch and choose explicit fallback mode
$PYTHON scripts/run_tests_safe.py \
  --no-continue-on-failure --launcher=prlimit
```

`--batch-mode=all` is available when a test suite relies on pytest state shared
across files, but it deliberately gives up per-file failure isolation and
forces effective concurrency to one (there is only one batch); that scope
still has the configured resource limits. For file batches, ordinary failures
follow `--continue-on-failure`; resource-limit and cleanup failures stop new
batches from being dispatched. Already-running batches continue under their
own timeout and resource limits. An external interrupt instead enters the
runner's bounded cleanup window.

Each run creates `outputs/test-runs/<timestamp>/` (or the path supplied by
`--output-dir`):

```text
metadata.json              # limits, concurrency plan, Git state, exact argv, and batch states
logs/batch-*.log            # pytest output and, for systemd, diagnostic journal tail
junit/batch-*.xml           # one JUnit artifact per batch
```

`--max-file-size-mb` defaults to `256` and is an `RLIMIT_FSIZE` per-regular-file
cap inherited by the batch and its descendants. It is a guard against runaway
artifacts, not a total output-directory quota; raise it when a test intentionally
writes a larger video, model, or coverage artifact.

Batch states are `PASSED`, `TEST_FAILURE`, `TIMEOUT`, `OOM`,
`RESOURCE_KILL`, `FILE_SIZE_LIMIT`, `CLEANUP_FAILURE`, `LAUNCH_FAILURE`, or
`INTERRUPTED`. `RESOURCE_KILL` means the process was SIGKILLed without reliable
evidence distinguishing an OOM from timeout escalation. `FILE_SIZE_LIMIT` means
Linux `RLIMIT_FSIZE` (`--max-file-size-mb`) was reached: it caps the size of
each regular file written by the batch process or its descendants (including
JUnit, coverage, videos, and caches), not aggregate log output or disk usage,
and may terminate the writer with `SIGXFSZ`. The recorded command and target
list in `metadata.json` make an individual batch reproducible. Exit code `0`
means every batch passed; `1` means a test/launch/cleanup/resource/file-size
failure; `2` means timeout or confirmed OOM; and `130` means the run was
interrupted. Resource and cleanup failures stop dispatching new batches; batches
already running finish under their own bounds. Ordinary test/launch failures
follow `--continue-on-failure`. CPU and RAM limits do not cap GPU VRAM, so use
`--cuda-visible-devices` and avoid concurrent GPU-heavy runs when needed.

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
| `[+]rounds=N` | int | 1 | Number of demo rounds to run |
| `[+]use_input=true` | bool | false | Pause before every step, including warmup (press Enter to continue) |
| `[+]max_updates=N` | int | 600 | Maximum public `TaskRunner.update()` calls per round; macro-boundary internal controller updates are limited separately |
| `[+]perf_count=true` | bool | false | Capture observations each step for performance analysis |
| `[+]print_updates=false` | bool | true | Disable reset/step `TaskUpdate` dumps while retaining summaries |
| `env.batch_size=N` | int | (from config) | Override the number of parallel environments |
| `task.seed=N` | int | (from config) | Override the randomization seed |
| `+env.viewer.disable=true` | bool | false | Run headless (no viewer window) |
| `env.hide_operators_in_camera=true` | bool | false | Exclude configured operators from native MuJoCo RGB/depth/mask rendering without changing physics |
| `[+]execution.update_boundary=...` | enum | `control_tick` | Public `update()` boundary: `control_tick`, `primitive`, `keypoint`, or `stage` |
| `[+]execution.render_internal_updates=false` | bool | true | Keep internal physics but refresh the viewer only once at each public boundary; boundary refreshes do not apply `step_delay` |
| `[+]execution.max_internal_updates_per_update=N` | int | 10000 | Per-environment controller-update limit within one public `update()` |
| `[+]execution.interval_selection...` | mapping | unset | Run between states immediately before or after configured `stage` / `phase` / `waypoint` keypoints |
| `[+]execution.interval_selection.{start,stop}.side=...` | enum | `before` / `after` | Endpoint side relative to its keypoint; the start default is `before`, while the stop default is `after` |
| `[+]execution.interval_selection.max_fast_forward_updates=N` | int | 10000 | Per-environment controller-update limit while `reset()` advances to the interval start boundary |

Any key present in the YAML config can be overridden on the command line following Hydra syntax:

Use `+key=value` when the selected YAML does not define the key, and
`key=value` when it already exists. `[+]` in the table means the prefix depends
on the selected config; using `+` for an existing key causes a Hydra composition
error.

```bash
# Multiple overrides
aao-demo --config-name stack_color_blocks +rounds=3 env.batch_size=4 +max_updates=500

# Override a nested key
aao-demo task.stages.0.param.pre_move.0.position="[0.4, 0.0, 0.1]"
```

Make each public update complete one YAML waypoint, beginning immediately
before the pick retract and ending immediately after the place retract:

```bash
aao-demo --config-name pick_and_place \
  +execution.update_boundary=keypoint \
  +execution.render_internal_updates=false \
  +execution.interval_selection.start.stage=pick_source \
  +execution.interval_selection.start.phase=post_move \
  +execution.interval_selection.start.waypoint=0 \
  +execution.interval_selection.start.side=before \
  +execution.interval_selection.stop.stage=place_source \
  +execution.interval_selection.stop.phase=post_move \
  +execution.interval_selection.stop.waypoint=0 \
  +execution.interval_selection.stop.side=after
```

The shipped `pick_and_place` config leaves this example commented out, so the
command adds the paths with `+`. When a selected config already defines a
path, override it without `+`. The public update boundary choices are:

- `control_tick`: return after one controller update; this default preserves
  the previous behavior.
- `primitive`: complete one runtime primitive. Arc sub-actions are separate
  primitive boundaries.
- `keypoint`: complete one YAML waypoint. An arc waypoint returns only after
  all of its sub-actions complete.
- `stage`: complete one whole stage, including its semantic condition checks.

`before` is the state before a referenced keypoint executes; neither its
action nor a condition bound to its completion has run. `after` is the state
after the entire keypoint and its completion-bound condition finish. The
explicit sides above match their defaults and make the command's intent
clear.

Configs written before `side` was available effectively started at
`after`. They still parse without the field, but the new start default is
`before`; add `start.side=after` when preserving the old reset behavior.

An interval stop always takes priority over a coarser boundary, so `stage`
cannot advance past a stop boundary in the middle of that stage. The public
update and reset fast-forward limits are independent; both default to `10000`.
With `execution.render_internal_updates=false`, all of those internal updates
still run, but their viewer refreshes and `step_delay` calls are coalesced into
one delay-free refresh at the public boundary.
See [Stages & Waypoints](../task-configuration/stages_and_waypoints.md#task-interval-boundary-selection)
for endpoint semantics and reporting.

`PolicyEvaluator` / `aao-eval` accepts only the default `control_tick` boundary
and rejects `execution.interval_selection` and
`execution.render_internal_updates=false`. An external policy must supply a
new action at every control tick, so the evaluator cannot synthesize the
intermediate actions required by the TaskRunner-only execution modes.

### Output

Each run writes a `summary.json` to the Hydra output directory
(`outputs/<date>/<time>/summary.json`) containing per-round success rates,
completion steps, timing, and failure reasons. `updates_used` includes the
untimed warmup update; `timed_updates` and `loop_frequency_hz` exclude it.
Timing covers only update execution, not interactive waits or console output.

## aao-unidoor-sweep

Run the UniDoor door/handle product space as a strictly serial Hydra matrix and
write a machine-readable result for every expected combination. IDs come from
the component index declared by the scene asset package, so the tested matrix
and the assets loaded by the task have one source of truth.

```bash
# All 55 doors x 47 handles (2,585 jobs)
aao-unidoor-sweep

# A smaller Cartesian product, in the specified order
aao-unidoor-sweep \
  --doors D001,D002 \
  --handles H001,H004,HL001
```

Hydra's basic launcher is serial, but it retains each job in the same Python
process. The wrapper therefore starts bounded multiruns (six jobs per process
by default) and waits for each one before starting the next. This keeps the
combination order serial while releasing simulator resources between batches.
Change the bound with `--launcher-batch-size`; do not use Hydra's
`hydra.sweeper.max_batch_size=1`, because an exceptional job can then prevent
later batches from running.

Every job uses `env.batch_size=1`, the configured task seed (42 by default),
and a disabled viewer. It preserves the task's cameras, sensors, timeouts,
control rates, and callback behavior. A full catalog run takes hours; use a
subset first when validating a new task configuration.

### Sweep outputs and failure records

The default root is `outputs/unidoor-sweeps/<timestamp>/`:

```text
sweep_manifest.json   # expected jobs, exact argv, catalog hashes, Git state
sweep.log             # combined stdout/stderr from every Hydra batch
report.json           # one structured result per expected combination
failures.csv          # only non-success combinations, with reproduction commands
batches/...           # Hydra configs, demo.log, and per-job summary.json files
```

`report.json` and `failures.csv` distinguish five states:

| State | Meaning |
|---|---|
| `SUCCESS` | Every recorded round and environment succeeded |
| `TASK_FAILURE` | The simulation completed and wrote a valid task failure |
| `NO_SUMMARY` | Hydra created the job directory but no `summary.json` was written |
| `NOT_STARTED` | The manifest expected the job, but Hydra never created its directory |
| `INVALID_SUMMARY` | `summary.json` exists but cannot be parsed or has no valid rounds |

A task-level failure does not make `aao-demo` itself return nonzero, so the
sweep always classifies the summaries rather than relying on Hydra's return
code. Each failed row includes a standalone `reproduce_command` with the exact
door, handle, seed, rounds, and update limit.

Rebuild the reports without starting simulations, or resume only combinations
with missing/invalid results in versioned attempt directories:

```bash
aao-unidoor-sweep --report outputs/unidoor-sweeps/20260827-180000
aao-unidoor-sweep --resume outputs/unidoor-sweeps/20260827-180000
```

Exit code `0` means every combination succeeded, `1` means only task-level
failures occurred, `2` means at least one job was not started or produced no
valid summary, and `130` means the sweep was interrupted. Reports are written
before returning any of these nonzero codes.

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
| `use_input=true` | bool | false | Pause before every step, including warmup |
| `get_obs=true` | bool | false | Call `capture_observation()` and pass to policy each step |
| `print_updates=false` | bool | true | Disable reset/step `TaskUpdate` dumps while retaining summaries |

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

For each task it reports the task name, the **operating subject** (the operator
that performs the stages and the robot model it is embodied as), the objects it
manipulates, the operations it performs, and a workflow generated from the
ordered stages.

```bash
aao-info                    # list every runnable task
aao-info pick_and_place     # a single config by exact name
aao-info 'open_door*'       # glob over config names (quote so the shell doesn't expand it)
aao-info -o press           # only tasks that press something
aao-info --object cup       # only tasks involving a "cup" object
aao-info -r airbot          # only tasks running on an airbot robot
aao-info -o pick -r p7      # combine filters (AND across categories)
aao-info --json             # machine-readable output
aao-info --verbose          # also report configs skipped as non-tasks
```

### Filtering

| Argument | Description |
|---|---|
| `PATTERN...` | Glob pattern(s) (`fnmatch`) matched against config names; an exact name matches itself. Default: all runnable tasks |
| `-o, --operation OP` | Keep tasks that use operation `OP` (repeatable, or comma-separated: `-o pick,place`) |
| `-b, --object OBJ` | Keep tasks referencing an object whose name contains `OBJ` (case-insensitive substring) |
| `-s, --scene GLOB` | Keep tasks whose `scene_name` matches the glob |
| `-r, --robot MODEL` | Keep tasks whose robot model contains `MODEL` (case-insensitive substring) |
| `--vocab`, `--keywords` | Aggregate all fields into a keyword vocabulary instead of a per-task report (see below) |
| `--json` | Emit a JSON array (or, with `--vocab`, a `{field: [values]}` object) instead of readable text |
| `--config-dir DIR` | Config directory (default: `./aao_configs`) |
| `--verbose` | Print configs skipped as non-tasks or on composition errors (to stderr) |
| `--no-progress` | Disable the progress line (see below) |

Filter categories are AND-combined; values within a category are OR-combined
(e.g. `-o pick -o place` keeps tasks that use pick **or** place). Name globs are
matched before composition, so filtering by name is cheap.

> **Progress:** each config must be composed by Hydra to decide whether it is a
> task, which takes a moment when there are many configs. While it works,
> `aao-info` shows a transient `Composing configs [i/total]` line on **stderr**.
> It is auto-enabled only when stderr is a terminal (so piped or redirected
> output stays clean) and can be turned off with `--no-progress`. Because it is
> on stderr, it never contaminates the text or `--json` output on stdout.

Example output:

```
Runnable tasks (40):

press_three_buttons
  operators:  arm (robotiq)
  objects:    button_blue, button_green, button_pink
  operations: press
  workflow:
    1. press button_blue [press_blue]
    2. press button_green [press_green]
    3. press button_pink [press_pink]
```

The **operating subject** comes from the task's operators (the `operator` a
stage runs on, plus any declared in `task_operators` / `env.operators`), each
annotated with its robot model — the `env.scene.layers[kind=mjcf]` XML stem (e.g.
`robotiq`, `airbot_play_with_g2p`). The model is shown inline when the scene
loads a single robot; when the scene loads several (or none, e.g. the mock
backend), a separate `robots:` line lists them and the inline model is omitted.
In `--json` output these are the `operators` (list of `{name, model}`) and
`robots` fields.

Objects and operations are cross-checked: the report prefers the declared
`env.mask_objects` / `env.operations`, and adds a `note:` line when they differ
from the objects/operations actually referenced by the stages.

### Vocabulary mode (`--vocab`)

`--vocab` (alias `--keywords`) flips the output from per-task to per-field: it
collapses all matching tasks into one glossary, where each field maps to the
sorted, de-duplicated union of its values. This is a controlled vocabulary an
agent (or a human) can search against for intelligent retrieval — "which
operations exist?", "what objects can be manipulated?", "which robot models are
available?".

```bash
aao-info --vocab              # glossary across all tasks
aao-info -r airbot --vocab    # glossary restricted to airbot tasks
aao-info --vocab --json       # {field: [sorted values]} for programmatic use
```

The aggregated fields are `configs`, `scenes`, `operators`, `robots`,
`objects`, `operations`, and `stage_names`. All active filters apply first, so
the vocabulary always reflects exactly the task subset you selected. Unresolved
interpolation placeholders (e.g. `${object_name}` from template configs) are
dropped so the vocabulary stays clean. Example:

```
Task vocabulary (40 tasks):

operators (3):
  arm, arm_a, observer

robots (9):
  airbot_play_with_g2, airbot_play_with_g2p, p7_arm_v3_with_umi_gripper_v3,
  p7_arm_with_g2p, p7_arm_with_xf9600, panda_robotiq, robotiq,
  umi_gripper_v3_mocap, xf9600_mocap

operations (6):
  move, pick, place, press, pull, push
```

## Config resolution

`aao-demo` and `aao-eval` resolve Hydra configs from `./aao_configs/` relative to the current working directory, and `aao-info` scans the same directory. Run them from the project root.
