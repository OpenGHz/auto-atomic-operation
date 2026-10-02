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

Batch states are `PASSED`, `NO_TESTS`, `TEST_FAILURE`, `TIMEOUT`, `OOM`,
`RESOURCE_KILL`, `FILE_SIZE_LIMIT`, `CLEANUP_FAILURE`, `LAUNCH_FAILURE`, or
`INTERRUPTED`. `NO_TESTS` means pytest returned its collection code 5 and the
JUnit report confirms a clean zero-test suite; it is recorded for transparency
but does not fail the overall run. A return code 5 without that evidence remains
`TEST_FAILURE`. `RESOURCE_KILL` means the process was SIGKILLed without reliable
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
aao-demo                                          # default task: pick_and_place
aao-demo task=cup_on_coaster                      # any option of aao_configs/task/
aao-demo task=cup_on_coaster embodiment=p7_g2p    # the same task on another robot
aao-demo task=cup_on_coaster render=gs            # Gaussian-splatting rendering
aao-demo task=pick_and_place backend=warp         # GPU (mujoco_warp) backend
```

To discover which task × embodiment combinations are runnable, use
[`aao-info`](#aao-info); it prints the exact `aao-demo` command for each.

### Config groups

Every run composes the single primary config `aao_configs/config.yaml`. A run
is selected by choosing config-group options, not by naming a config file —
`--config-name <task>` no longer exists. Group selections need no `+` prefix.

| Group | Options | Default | Description |
|---|---|---|---|
| `task` | `aao_configs/task/*.yaml` | `pick_and_place` | Stages, waypoints, and randomization. Each task also selects its scene and default embodiment |
| `embodiment` | `aao_configs/embodiment/*.yaml` (e.g. `robotiq_mocap`, `xf9600_mocap`, `franka_robotiq`, `p7_g2p`, `airbot_play_g2p`) | set by the task | Robot and gripper. Task-specific tuning in `adapt/<task>/<embodiment>.yaml` is applied automatically |
| `render` | `mujoco`, `gs` | `mujoco` | `gs` enables Gaussian-splatting rendering; choose the background with `render_assets/background=<name>` |
| `backend` | `cpu`, `warp` | `cpu` | `backend=warp` selects the GPU mujoco_warp backend |
| `platform` | `egl` | none | `platform=egl` sets the EGL/NVIDIA environment variables for headless rendering |
| `observation` | `default`, `rgb_only` | `default` | Camera resolution and modalities; tune single fields with `observation.camera.width=...`, `observation.camera.enable_mask=true`, ... |
| `camera_layout` | `all`, `operator_only`, `no_operator` | `all` | Which camera roles (wrist / scene / object) are kept |
| `execution` | `physical`, `object_only` | `physical` | `object_only` transports stage objects kinematically without operators |
| `scene` | `aao_configs/scene/*.yaml` | set by the task | Rarely overridden directly |
| `simulator` | `mujoco`, `mock` | `mujoco` | Mock tasks select `mock` themselves |

### Hydra overrides

| Override | Type | Default | Description |
|---|---|---|---|
| `[+]rounds=N` | int | 1 | Number of demo rounds to run |
| `[+]use_input=true` | bool | false | Pause before every step, including warmup (press Enter to continue) |
| `[+]max_updates=N` | int | 600 | Maximum public `TaskRunner.update()` calls per round; macro-boundary internal controller updates are limited separately |
| `[+]perf_count=true` | bool | false | Capture observations each step for performance analysis |
| `[+]print_updates=false` | bool | false | Disable reset/step `TaskUpdate` dumps while retaining summaries |
| `env.batch_size=N` | int | (from config) | Override the number of parallel environments |
| `task.seed=N` | int \| null | (from config) | Fix the run seed. Unset (`null`) keeps the run random but logs the concrete seed, so `task.seed=<logged value>` replays it; `0` is a normal seed |
| `+env.viewer.disable=true` | bool | false | Run headless (no viewer window) |
| `env.hide_operators_in_camera=true` | bool | false | Exclude configured operators from native MuJoCo RGB/depth/mask rendering without changing physics |
| `[+]execution.update_boundary=...` | enum | `control_tick` | Public `update()` boundary: `control_tick`, `primitive`, `keypoint`, or `stage` |
| `[+]execution.render_internal_updates=false` | bool | true | Keep internal physics but refresh the viewer only once at each public boundary; boundary refreshes do not apply `step_delay` |
| `[+]execution.max_internal_updates_per_update=N` | int | 10000 | Per-environment controller-update limit within one public `update()` |
| `[+]execution.interval_selection...` | mapping | unset | Run between states immediately before or after configured `stage` / `phase` / `waypoint` keypoints |
| `[+]execution.interval_selection.{start,stop}.side=...` | enum | `before` / `after` | Endpoint side relative to its keypoint; the start default is `before`, while the stop default is `after` |
| `[+]execution.interval_selection.max_fast_forward_updates=N` | int | 10000 | Per-environment controller-update limit while `reset()` advances to the interval start boundary |
| `[+]execution.keypoint_selection=[...]` | list | unset | Ordered keypoints to execute: task-wide ordinals or `{stage, phase?, waypoint?}` entries, with negative indexes counting from the end; unlisted keypoints are skipped. Mutually exclusive with `interval_selection` |

Any key present in the composed config can be overridden on the command line following Hydra syntax:

Use `+key=value` when the composed config does not define the key, and
`key=value` when it already exists. `[+]` in the table means the prefix depends
on the selected task; using `+` for an existing key causes a Hydra composition
error.

```bash
# Multiple overrides
aao-demo task=stack_color_blocks +rounds=3 env.batch_size=4 +max_updates=500

# Override a nested key
aao-demo task.stages.0.param.pre_move.0.position="[0.4, 0.0, 0.1]"
```

Make each public update complete one YAML waypoint, beginning immediately
before the pick retract and ending immediately after the place retract:

```bash
aao-demo task=pick_and_place \
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

The shipped `task/pick_and_place.yaml` leaves this example commented out, so the
command adds the paths with `+`. When the selected task already defines a
path, override it without `+`.

Run only some keypoints, skipping the rest, by replacing the contiguous
interval with a keypoint selection. Each entry is either a task-wide keypoint
ordinal (negative counts from the end) or a `stage` / `phase` / `waypoint`
scope:

```bash
aao-demo task=place_blocks_on_disk \
  ~execution.interval_selection \
  +execution.keypoint_selection="[0, -1]"
```

```bash
aao-demo task=place_blocks_on_disk \
  ~execution.interval_selection \
  +execution.keypoint_selection="[{stage: pick_cube_yellow_2}, {stage: place_cube_orange_3_in_disk, waypoint: -1}]"
```

Hydra expands a list only when the whole path is assigned, so a keypoint
selection is overridden as one `[...]` value rather than entry by entry.

The public update boundary choices are:

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
Its `run_config` block records the run name (`task__embodiment[__render]`,
e.g. `pick_and_place__robotiq_mocap` or `press_blue_button__p7_g2p__gs`; the
render part is omitted for native MuJoCo) and the selected config-group
`choices`.

## aao-unidoor-sweep

Run the UniDoor door/handle product space as a bounded Hydra matrix and write a
machine-readable result for every expected combination. IDs come from the
component index declared by the scene asset package, so the tested matrix and
the assets loaded by the task have one source of truth.

```bash
# All 55 doors x 47 handles (2,585 jobs), using the default
# task=open_door_unidoor embodiment=p7_v4_umi_v3
aao-unidoor-sweep

# Select the legacy P7 V3 embodiment explicitly
aao-unidoor-sweep --embodiment p7_v3_umi_v3

# A smaller Cartesian product, in the specified order
aao-unidoor-sweep \
  --doors D001,D002 \
  --handles H001,H004,HL001

# Start from all catalog assets and exclude selected IDs
aao-unidoor-sweep \
  --exclude-doors D003,D007 \
  --exclude-handles H002,HL016

# Hold each handle fixed and traverse all selected doors before the next handle
aao-unidoor-sweep --traversal-order handle-first

# Quick cross-section: first door x all handles, then first handle x other doors
aao-unidoor-sweep --simple-test

# Choose the two anchors explicitly
aao-unidoor-sweep \
  --simple-test \
  --simple-test-door D003 \
  --simple-test-handle H007
```

`--exclude-doors` and `--exclude-handles` are applied after the positive
`--doors`/`--handles` selection. Therefore they can either subtract from the
whole catalog or from an explicit subset. Unknown or duplicate excluded IDs,
and exclusions that leave no door or no handle, are rejected before launch.

`--traversal-order door-first` is the default: each door is paired with all
handles before moving to the next door. Use `handle-first` to invert the loops.
The selected order is stored in the manifest and retained by resume runs.

`--simple-test` avoids the full Cartesian product. It runs one selected door
against every selected handle, then one selected handle against every remaining
selected door, for `door_count + handle_count - 1` combinations. The anchors
default to the first selected door and handle; `--simple-test-door` and
`--simple-test-handle` override them. Positive selections and exclusions are
applied first, so they can narrow what “every” means.

By default, Hydra's Joblib launcher runs up to four combinations concurrently.
Change the bound with `--max-concurrency`; set it to `1` to use Hydra's basic
serial launcher. `--launcher-batch-size` separately caps the combinations held
by one Hydra process (six by default).

Failure stopping operates between parallel waves: all combinations already
launched in the current wave finish, their summaries are recorded, and no next
wave starts if any result failed. Disable this behavior with
`--no-stop-on-failure` to continue through later batches. Resume skips the
successful peers from a failed wave and retries only failed or unstarted jobs.

Every job uses `env.batch_size=1`, the configured task seed (42 by default),
and a disabled viewer. It preserves the task's cameras, sensors, timeouts,
control rates, and callback behavior. A full catalog run takes hours; use a
subset first when validating a new task configuration.

The terminal shows one compact progress bar by default. Hydra, MuJoCo, and task
execution output is retained in `sweep.log`; pass `--verbose` to stream it live
for debugging. At exit the CLI prints the output directory containing all
artifacts.

### Sweep outputs and failure records

The default root is `outputs/unidoor-sweeps/<timestamp>/`:

```text
sweep_manifest.json   # expected jobs, attempts, cursor, exact argv, catalog/Git state
sweep.log             # combined stdout/stderr from every Hydra batch
report.json           # one structured result per expected combination
failures.csv          # only non-success combinations, with reproduction commands
batches/...           # Hydra configs, demo.log, and per-job summary.json files
```

`report.json` and `failures.csv` distinguish these states:

| State | Meaning |
|---|---|
| `SUCCESS` | Every recorded round and environment succeeded |
| `TASK_FAILURE` | The simulation completed and wrote a valid task failure |
| `NO_SUMMARY` | Hydra created the job directory but no `summary.json` was written |
| `NOT_STARTED` | The manifest expected the job, but Hydra never created its directory |
| `INVALID_SUMMARY` | `summary.json` exists but cannot be parsed or has no valid rounds |
| `LAUNCHER_FAILURE` | Hydra itself exited nonzero before a valid combination result was accepted |

A task-level failure does not make `aao-demo` itself return nonzero, so the
sweep always classifies the summaries rather than relying on Hydra's return
code. Each failed row includes a standalone `reproduce_command` with the exact
door, handle, seed, rounds, and update limit.

On failure, `sweep_manifest.json.progress` records the failed `job_num`, door,
handle, status, reason, and next resume cursor. Jobs after the cursor stay
`PENDING`. After fixing the task, resume the same directory; the failed
combination is retried first, successful earlier combinations are not repeated,
and each retry is preserved below a versioned `resume/<attempt>/` directory:

```bash
aao-unidoor-sweep --report outputs/unidoor-sweeps/20260827-180000
aao-unidoor-sweep --resume outputs/unidoor-sweeps/20260827-180000
aao-unidoor-sweep --resume-latest
```

Use `--report` when only rebuilding `report.json` and `failures.csv`. A resume
also retries task-level failures, missing/invalid summaries, launcher failures,
and combinations that never started. `--resume-latest` selects the valid sweep
under `outputs/unidoor-sweeps/` whose manifest was updated most recently;
explicit `--resume` remains available when reproducing an older run.
Manifests written before the config-group layout (they record a
`config_name` instead of `task` / `embodiment`) can still be reported, but
not resumed; start a new sweep with `--task` / `--embodiment`.

Exit code `0` means every combination succeeded. A strict stop returns `1` for
a task-level failure or `2` for an infrastructure/launcher failure; later
`PENDING` combinations do not change that diagnosis. Exit code `130` means the
sweep was interrupted. Reports are written before returning any nonzero code.

## aao-eval

Run policy evaluation. Same Hydra config system as `aao-demo` but accepts an external policy.

```bash
aao-eval task=pick_and_place       # evaluate with ConfigDrivenDemoPolicy (default)
aao-eval task=policy_eval_mock     # mock backend evaluation
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

Provide a `policy` section in the task YAML (or pass `+policy._target_=...` on the command line) to use a custom policy:

```yaml
policy:
  _target_: my_package.MyPolicy
  checkpoint: /path/to/model.pt
```

When `policy` is omitted, `aao-eval` defaults to `auto_atom.ConfigDrivenDemoPolicy`, which replays the same primitive actions that `aao-demo` uses. See [Policy Evaluation](../tools/policy_evaluation.md) for the full API reference.

## aao-info

Introspect the **runnable task variants** in `aao_configs/task/`. Each option
of the `task` group (`_`-prefixed fragments are skipped) is reported once for
its default embodiment and once more for every embodiment it has a dedicated
adaptation for (`aao_configs/adapt/<task>/<embodiment>.yaml`) — the validated
task × embodiment combinations. A variant is kept only when it composes into a
real task, i.e. with a non-empty `task.stages` after Hydra composition.

For each variant it reports the `aao-demo` command that selects it, the
embodiment (marked `(default)` when it is the task's own default), the
available render modes (`mujoco | gs` when the scene has GS assets), the
**operating subject** (the operator that performs the stages and the robot
model it is embodied as), the objects it manipulates, the operations it
performs, and a workflow generated from the ordered stages.

```bash
aao-info                    # list every runnable task variant
aao-info pick_and_place     # one task (all of its embodiment variants)
aao-info 'open_door*'       # glob over task names (quote so the shell doesn't expand it)
aao-info -o press           # only tasks that press something
aao-info --object cup       # only tasks involving a "cup" object
aao-info -r airbot          # only variants running on an airbot embodiment
aao-info -o pick -r p7      # combine filters (AND across categories)
aao-info --json             # machine-readable output
aao-info --verbose          # also report variants skipped as non-tasks
```

### Filtering

| Argument | Description |
|---|---|
| `PATTERN...` | Glob pattern(s) (`fnmatch`) matched against task names; an exact name matches itself. Default: all runnable tasks |
| `-o, --operation OP` | Keep tasks that use operation `OP` (repeatable, or comma-separated: `-o pick,place`) |
| `-b, --object OBJ` | Keep tasks referencing an object whose name contains `OBJ` (case-insensitive substring) |
| `-s, --scene GLOB` | Keep tasks whose `scene_name` matches the glob |
| `-r, --robot MODEL` | Keep variants whose embodiment name or robot model contains `MODEL` (case-insensitive substring) |
| `--vocab`, `--keywords` | Aggregate all fields into a keyword vocabulary instead of a per-task report (see below) |
| `--json` | Emit a JSON array (or, with `--vocab`, a `{field: [values]}` object) instead of readable text |
| `--config-dir DIR` | Config directory (default: `./aao_configs`) |
| `--verbose` | Print variants skipped as non-tasks or on composition errors (to stderr) |
| `--no-progress` | Disable the progress line (see below) |

Filter categories are AND-combined; values within a category are OR-combined
(e.g. `-o pick -o place` keeps tasks that use pick **or** place). Name globs are
matched before composition, so filtering by name is cheap.

> **Progress:** each variant must be composed by Hydra to decide whether it is a
> task, which takes a moment when there are many variants. While it works,
> `aao-info` shows a transient `Composing configs [i/total]` line on **stderr**.
> It is auto-enabled only when stderr is a terminal (so piped or redirected
> output stays clean) and can be turned off with `--no-progress`. Because it is
> on stderr, it never contaminates the text or `--json` output on stdout.

Example output:

```
Runnable task variants (37):

...

press_blue_button  (scene: press_three_buttons)
  run:        aao-demo task=press_blue_button
  embodiment: robotiq_mocap (default)
  render:     mujoco | gs
  operators:  arm (robotiq)
  objects:    button_blue
  operations: press
  workflow:
    1. press button_blue [press_button_blue]

press_blue_button embodiment=p7_g2p  (scene: press_three_buttons)
  run:        aao-demo task=press_blue_button embodiment=p7_g2p
  embodiment: p7_g2p
  render:     mujoco | gs
  operators:  arm (p7_arm_with_g2p)
  objects:    button_blue
  operations: press
  workflow:
    1. press button_blue [press_blue_button]
```

The `(scene: ...)` suffix appears when the scene differs from the task name.
In `--json` output each variant carries `task`, `embodiment`,
`default_embodiment`, `scene_name`, `renders`, and the `overrides` list that
selects it.

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

The aggregated fields are `tasks`, `embodiments`, `scenes`, `operators`,
`robots`, `objects`, `operations`, and `stage_names`. All active filters apply first, so
the vocabulary always reflects exactly the task subset you selected. Unresolved
interpolation placeholders (e.g. `${object_name}` from template configs) are
dropped so the vocabulary stays clean. Example:

```
Task vocabulary (23 tasks):

embodiments (10):
  airbot_play_g2, airbot_play_g2p, franka_robotiq, p7_g2p, p7_v3_umi_v3,
  p7_v4_umi_v3, p7_xf9600, robotiq_mocap, umi_v3_mocap, xf9600_mocap

operators (3):
  arm, arm_a, observer

operations (6):
  move, pick, place, press, pull, push
```

## Config resolution

`aao-demo` and `aao-eval` compose `./aao_configs/config.yaml` relative to the current working directory, and `aao-info` scans the same directory. Run them from the project root.

`config.yaml` composes its groups in this order, later entries winning and
command-line overrides winning over everything: `simulator`, `execution`,
`observation`, `camera_layout`, `scene`, `embodiment`, `task`, `render`, the
optional `render_assets/{embodiment,scene,task}: ${render}/<name>` asset
bindings, the optional `adapt: ${task}/${embodiment}` tuning, `backend`, and
`platform`. A task picks its scene and default embodiment with
`override /scene:` / `override /embodiment:` in its own defaults list, so
`embodiment=...` on the command line still takes priority. `render`,
`backend`, and `platform` compose after the task and are therefore user-level
choices a task cannot override. See
[Reusing & Creating Tasks](../task-configuration/reusing_and_creating_tasks.md)
for how to add tasks and robots, and the
[config-groups migration note](../migration-notes/aao_configs_config_groups.md)
for the mapping from the old `--config-name` names.
