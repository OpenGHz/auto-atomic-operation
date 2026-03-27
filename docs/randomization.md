# Pose Randomization

auto_atom supports per-entity pose randomization applied at each `reset()`.
This is used to evaluate task robustness under varying initial conditions.

## YAML Configuration

Add a `randomization` block under `task` in your YAML config.
Keys are object or operator names; values define per-axis offset ranges.

```yaml
task:
  seed: 42                     # numpy RNG seed for reproducibility
  # randomization_debug: true  # see "Debug Mode" below
  randomization:
    source_block:
      x: [-0.03, 0.03]        # metres, world frame
      y: [-0.03, 0.03]
      # yaw: [-0.524, 0.524]  # radians
      collision_radius: 0.04   # metres, for collision rejection
    arm:
      x: [-0.015, 0.015]
      y: [-0.015, 0.015]
      collision_radius: 0.15
```

### Supported axes

| Axis    | Unit    | Description                         |
|---------|---------|-------------------------------------|
| `x`     | metres  | World X displacement                |
| `y`     | metres  | World Y displacement                |
| `z`     | metres  | World Z displacement                |
| `roll`  | radians | Additive roll offset (about X)      |
| `pitch` | radians | Additive pitch offset (about Y)     |
| `yaw`   | radians | Additive yaw offset (about Z)       |

Each axis takes a `[min, max]` tuple. Omitted axes default to `[0, 0]` (no randomization).

### collision_radius

Each entity has a `collision_radius` (default 0.05 m). After sampling, pairwise
Euclidean distances are checked: if any two entities are closer than the sum of
their radii, the sample is rejected and redrawn. After 100 failed attempts the
last sample is applied with a warning.

## How It Works

The randomization logic lives in `SceneBackend` (the mixin used by `MujocoTaskBackend`).

### Lifecycle

1. **`setup()`** — calls `_record_default_poses()` to snapshot the canonical pose
   of every entity listed in `randomization` (after `env.reset()` and operator `home()`).

2. **`reset()`** — after restoring the scene to its canonical state:
   - Calls `_record_default_poses()` if not already recorded.
   - Calls `_apply_randomization()` which:
     1. Resolves each key to an object handler or operator handler.
     2. Samples a uniform random offset per axis within `[min, max]`.
     3. Applies the offset to the default pose (translation is additive;
        rotation is additive in RPY then converted back to quaternion).
     4. Checks pairwise collisions; resamples if needed.
     5. Calls `handler.set_pose(new_pose)` for each entity.
   - Refreshes the viewer.

3. **`TaskRunner.reset()`** — after the backend reset, collects the realized
   poses of all randomized entities and returns them in
   `TaskUpdate.details["initial_poses"]`. This allows the caller to log initial
   conditions without accessing backend internals.

### Entity resolution

Each randomization key is resolved in order:
1. `object_handlers[name]` — uses `get_pose()` / `set_pose()`.
2. `operator_handlers[name]` — uses `get_base_pose()` / `set_pose()`.
3. If neither matches, a warning is emitted and the key is skipped.

## Multi-Round Evaluation

Use the `rounds` top-level config key (default 1) to run the task multiple times
with different random seeds:

```bash
python examples/run_demo.py rounds=10
python examples/run_demo.py --config-name cup_on_coaster rounds=20
```

Each round resets the scene (applying a fresh random sample) and runs all stages.
A summary is printed at the end:

```
============================================================
SUMMARY
============================================================
Success rate: 8/10

  Round 1: [OK]
    source_block: pos=[0.0123, -0.0201, 0.06]
    arm: pos=[-0.005, 0.012, 0.0]
    stage pick_source: completed
    stage place_source: completed
  Round 2: [FAIL]
    source_block: pos=[-0.028, 0.015, 0.06]
    ...
============================================================
```

## Debug Mode

Set `randomization_debug: true` to cycle through extreme poses before random
sampling. The sequence is:

1. All entities at all-axis **minimum** simultaneously.
2. All entities at all-axis **maximum** simultaneously.
3. For each entity, for each non-trivial axis (where min != max):
   one case at axis min, one at axis max (others at default).

After exhausting all extreme cases, subsequent resets switch to normal random
sampling. This is useful for verifying that configured ranges don't cause
unreachable grasps or collisions.

```bash
python examples/run_demo.py task.randomization_debug=true rounds=20
```

## Reproducibility

Set `task.seed` to fix the numpy RNG seed:

```bash
python examples/run_demo.py task.seed=42 rounds=5
```

The same seed produces the same sequence of random poses across runs.
