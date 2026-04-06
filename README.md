<div align="center">

<h1>Auto Atomic Operation</h1>

[![PyPI](https://img.shields.io/pypi/v/auto-atomic-operation)](https://pypi.org/project/auto-atomic-operation/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A YAML-driven atomic operation framework for robotic manipulation.

</div>

`auto-atomic-operation` lets you define robotic manipulation tasks — move, grasp, release, pick, place, push, pull, press — as declarative YAML files. A built-in state machine handles task sequencing, pose resolution, end-effector control, and execution tracking. A plugin-based backend system decouples task logic from the underlying hardware or simulator, making it easy to run the same task definition against a real robot, a physics simulator, or a lightweight mock for testing.

## Features

- **Hydra-powered task configuration** — describe multi-stage manipulation tasks in YAML; full [Hydra](https://hydra.cc) support means `_target_` instantiation, variable interpolation, and command-line overrides work out of the box
- **Flexible pose references** — specify targets relative to world, robot base, end-effector, or tracked objects
- **Pluggable backends** — connect any robot or simulator by implementing a small set of abstract interfaces
- **Pose randomization** — per-object position/orientation randomization with automatic collision avoidance on reset
- **Multi-arm support** — single-arm and dual-arm (left/right) topologies
- **Execution records** — detailed per-stage status, failure reasons, and timing after every run
- **MuJoCo backend included** — a ready-to-use backend with RGB-D cameras, tactile sensors, force/torque, IMU, and joint state support

## Installation

Requires **Python 3.10+**.

### Install from PyPI

```bash
pip install auto-atomic-operation
```

Note: the PyPI release may lag behind the latest source version. You can check the README shown on the [PyPI project page](https://pypi.org/project/auto-atomic-operation/) for the documentation corresponding to that published release.

The PyPI version is relatively stable, but it may not include some newer features, and it does not ship with demos. If you want to use the latest features and demos, install from source instead.

### Install from source

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/OpenGHz/auto-atomic-operation.git
```

Note: the mujoco demo assets are stored with Git LFS, so the above command skips downloading them. If you want to use the demos, follow the instructions in the [Examples](#examples) section below to download the assets separately.

Then install the package:

```bash
# Core framework only
pip install -e .

# With the built-in MuJoCo backend
pip install -e ".[mujoco]"
```

## Examples

The runnable entry points live under [`auto_atom/runner/`](auto_atom/runner/).
At runtime, `aao_demo` and `aao_eval` look for configs in `./aao_configs/` relative to the current working directory:

```
auto_atom/
└── runner/
    ├── demo.py                   # `aao_demo`
    ├── policy_eval.py            # `aao_eval`
    └── common.py                 # shared runner loop / config loading

aao_configs/
├── pick_and_place.yaml        # Pick and place a block (default)
├── pick_and_place_franka.yaml # Pick and place a block with a Franka arm
├── cup_on_coaster.yaml        # Pick a cup and place it on a coaster
├── stack_color_blocks.yaml    # Stack three colored blocks
├── press_three_buttons.yaml   # Press three buttons in sequence
├── open_drawer.yaml           # Pull a drawer open
├── close_drawer.yaml          # Push a drawer closed
└── mock.yaml                  # Mock backend demo (no simulator required)
```

When using the package after `pip install`, make sure you run commands from a directory that contains `aao_configs/`, or create that folder there and place your task YAML files inside it.

### Mock example (no robot or simulator required)

```bash
aao_demo --config-name mock
```

Uses the in-memory mock backend — ideal for testing task logic in isolation.

### Policy evaluation example

```bash
aao_eval --config-name pick_and_place
```

Runs a policy-driven rollout evaluator that reuses the framework's stage success conditions and shared result types. When the config does not define `policy`, `aao_eval` defaults to `auto_atom.ConfigDrivenDemoPolicy`, so normal demo configs can be evaluated directly without creating a separate eval config.

### MuJoCo demos

#### Robotiq tasks

| | |
|:---:|:---:|
| ![pick_and_place](assets/videos/pick_and_place.gif) | ![cup_on_coaster](assets/videos/cup_on_coaster.gif) |
| `pick_and_place` | `cup_on_coaster` |
| ![stack_color_blocks](assets/videos/stack_color_blocks.gif) | ![press_three_buttons](assets/videos/press_three_buttons.gif) |
| `stack_color_blocks` | `press_three_buttons` |
| ![open_drawer](assets/videos/open_drawer.gif) | ![close_drawer](assets/videos/close_drawer.gif) |
| `open_drawer` | `close_drawer` |
| ![open_hinge_door](assets/videos/open_hinge_door.gif) | ![close_hinge_door](assets/videos/close_hinge_door.gif) |
| `open_hinge_door` | `close_hinge_door` |

#### Franka task

| |
|:---:|
| ![pick_and_place_franka](assets/videos/pick_and_place_franka.gif) |
| `pick_and_place_franka` |

Make sure to install Git LFS and pull the assets after cloning:

```bash
# For Debian/Ubuntu, you can install Git LFS via packagecloud.
# Firstly, you may need to add the repository to your system through the following sh script:
# curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh
sudo apt-get install git-lfs
git lfs pull
```

List all available demos:

```bash
aao_demo --list
```

Run any demo by passing its config name:

```bash
aao_demo --config-name <config>
```

Available configs:

Robotiq:

| Config | Description |
| ------ | ----------- |
| `pick_and_place` (default) | Pick a block and place it at a target location with the Robotiq gripper setup |
| `cup_on_coaster` | Pick a cup from a randomized position and place it on a coaster |
| `stack_color_blocks` | Stack three colored blocks: blue on orange, then yellow on blue |
| `press_three_buttons` | Press three buttons (blue, green, pink) in sequence |
| `open_drawer` | Pull a drawer open |
| `close_drawer` | Push a drawer closed (starts from the open position) |
| `open_hinge_door` | Pull a hinge door open |
| `close_hinge_door` | Push a hinge door closed (starts from the open position) |

Franka:

| Config | Description |
| ------ | ----------- |
| `pick_and_place_franka` | Pick a block and place it at a target location with a Franka arm |

Each demo runs in the MuJoCo physics simulator with RGB-D cameras, tactile sensors, and randomized object placement. The scene XML for each demo is at `assets/xmls/scenes/<config>/demo.xml` and can be previewed with:

```bash
(cd assets/xmls/scenes/<config>/ && python -m mujoco.viewer --mjcf demo.xml)
```

Config values can be overridden on the command line via Hydra:

```bash
aao_demo task.seed=0
aao_demo "task.randomization.source_block.x=[-0.05,0.05]"
```

### 3D GS Rendering Demos

Install 3D GS dependencies:
```bash
pip install -e ."[gs]"
# pip install "gaussian_renderer @ git+https://github.com/OpenGHz/GaussianRenderer.git"
```

Run the following command to download the assets for the 3D GS assets for the demos:

```bash
pip install huggingface_hub httpx[socks]
hf download OpenGHz/auto-atom-assets --repo-type=dataset --include "assets/gs/*" --local-dir .
```

If the download is slow, you may need to configure a terminal proxy first. You can also download the assets in your browser from https://huggingface.co/datasets/OpenGHz/auto-atom-assets/tree/main.

| | |
|:---:|:---:|
| ![cup_on_coaster_gs](assets/videos/cup_on_coaster_gs.gif) | ![stack_color_blocks_gs](assets/videos/stack_color_blocks_gs.gif) |
| `cup_on_coaster_gs` | `stack_color_blocks_gs` |
| ![press_three_buttons_gs](assets/videos/press_three_buttons_gs.gif) | ![arrange_flowers_gs](assets/videos/arrange_flowers_gs.gif) |
| `press_three_buttons_gs` | `arrange_flowers_gs` |
| ![hang_toothbrush_cup_gs](assets/videos/hang_toothbrush_cup_gs.gif) | ![wipe_the_table_gs](assets/videos/wipe_the_table_gs.gif) |
| `hang_toothbrush_cup_gs` | `wipe_the_table_gs` |

Configuration files related to 3D GS end with `_gs.yaml` and are run in the same way as described above.

## Tools

- **[Data Collection Guide](docs/data_collection.md)** — recording task demos (GIF/MP4) and comparing GS vs native MuJoCo rendering
- **[Execution Completion Flow](docs/execution_completion_flow.md)** — how `pre_move`, `eef`, and `post_move` decide they are done, how that feeds into stage success/failure, and a flowchart of the control path
- **[Policy Evaluation](docs/policy_evaluation.md)** — evaluate an external policy model with `PolicyEvaluator`, reuse `TaskUpdate` / `ExecutionRecord` / `ExecutionSummary`, and connect policy outputs to environment actions
- **[Tune Initial State](docs/tune_initial_state.md)** — interactive tkinter + MuJoCo viewer tool for tuning operator base pose, EEF pose, and gripper before writing the values into task YAML
- **[XML / Mesh / GS Migration Notes](docs/skills/xml_mesh_gs_migration_notes.md)** — if you want to migrate your own XML, mesh, or Gaussian assets into this project's normalized asset layout, use this as the reference checklist

## Quick Start

### 1. Define a task in YAML

```yaml
# task.yaml
backend: auto_atom.mock.build_mock_backend

task:
  env_name: my_env
  stages:
    - name: pick_cup
      object: cup
      operation: pick
      operator: arm_a
      param:
        pre_move:
          - position: [0.45, -0.10, 0.08]
            rotation: [0.0, 1.57, 0.0]
            reference: object_world
        eef:
          close: true

    - name: place_on_shelf
      object: shelf
      operation: place
      operator: arm_a
      param:
        pre_move:
          - position: [0.10, 0.25, 0.16]
            orientation: [0.0, 0.0, 0.0, 1.0]
            reference: world
        eef:
          close: false

operators:
  - name: arm_a
```

### 2. Run the task

```python
from pathlib import Path
from auto_atom.runtime import ComponentRegistry, TaskRunner

ComponentRegistry.clear()
runner = TaskRunner().from_yaml(Path("task.yaml"))

runner.reset()
while True:
    update = runner.update()
    if bool(update.done.all()):
        break

for record in runner.records:
    print(record)

runner.close()
```

## YAML Configuration Reference

Task files are processed by [Hydra](https://hydra.cc) / OmegaConf, so the full Hydra feature set is available: `_target_` instantiation, `${key}` interpolation, structured configs, and command-line overrides (`key=value`).

A task file has four top-level keys:

| Key         | Description                                                                           |
| ----------- | ------------------------------------------------------------------------------------- |
| `env`       | Hydra `_target_` instantiation of the batched environment, registered via `ComponentRegistry` |
| `backend`   | Dotted import path to the backend factory function                                    |
| `task`      | Task definition: `env_name`, `seed`, and a list of `stages`                           |
| `operators` | Named operators with assigned roles                                                   |

### Stage definition

```yaml
- name: <stage_name>          # Unique stage identifier
  object: <object_name>       # Target object
  operation: pick|place|push|pull|press|move|grasp|release
  operator: <operator_name>   # Which arm/robot executes this stage
  blocking: true              # Optional, default true
  param:
    pre_move:                 # Approach waypoints (list of pose configs)
      - position: [x, y, z]
        rotation: [rx, ry, rz]    # Euler angles (rad), or:
        orientation: [x, y, z, w] # Quaternion
        reference: world|base|eef|object|object_world|eef_world|auto
    post_move:                # Retreat waypoints (same format as pre_move)
    eef:
      close: true|false       # Gripper open/close
```

### Atomic operations

Each stage executes up to three sub-phases in order: **pre_move** (approach waypoints) → **eef** (gripper action) → **post_move** (retreat waypoints). The table below lists only the sub-phases that each operation **requires**; any unlisted sub-phases are optional and executed when provided.

Pre- and post-conditions constrain when an operation may run and what constitutes success. The timing of condition checks depends on the operation:

| Operation | Description | Required sub-phases | Pre-condition | Pre-condition checked | Post-condition | Post-condition checked |
| --------- | ----------- | ------------------- | ------------- | --------------------- | -------------- | ---------------------- |
| `move`    | Move to a target pose without interacting with any object | pre_move | — | — | `reached` | After pre_move |
| `grasp`   | Close the gripper at the current position to grasp an object | eef | `released` | Before eef | `grasped` | After eef |
| `release` | Open the gripper at the current position to release the held object | eef | `grasped` | Before eef | `released` | After eef |
| `pick`    | Approach the object, grasp it, then retreat | pre_move, eef | `released` | Before pre_move | `grasped` | After post_move |
| `place`   | Approach the target, release the object, then retreat | pre_move, eef | `grasped` | Before pre_move | `released` | After post_move |
| `push`    | Move to the object and push it to the target location | pre_move | — | — | `displaced` | After post_move |
| `pull`    | Move to the object, grasp it, then pull to the target | pre_move | `grasped` | After eef | `grasped` | After post_move |
| `press`   | Move to the object and press it at the target pose | pre_move | — | — | `contacted` | After eef |

**Condition constraints:**

| Constraint   | Meaning                                                                         |
| ------------ | ------------------------------------------------------------------------------- |
| `released`   | Operator is not currently holding any object                                    |
| `grasped`    | Operator is currently holding an object                                         |
| `contacted`  | Operator end-effector is in contact with the target object                      |
| `displaced`  | Target object has moved beyond a threshold distance from its original pose      |
| `reached`    | Operator end-effector is within tolerance of the stage's final target pose      |

For detailed implementation of these conditions in the MuJoCo backend, including detection logic and configurable thresholds, see [MuJoCo Backend Conditions](docs/mujoco_backend_conditions.md).

### Pose references

| Reference      | Description                            |
| -------------- | -------------------------------------- |
| `world`        | Fixed world frame                      |
| `base`         | Robot base frame                       |
| `eef`          | Current end-effector frame             |
| `object`       | Object frame (tracks object movement)  |
| `object_world` | Object position with world orientation |
| `eef_world`    | EEF position snapshot at command start |
| `auto`         | Automatically determined from context  |

## Architecture

```
auto_atom/
├── framework.py        # Pydantic configuration models (YAML schema)
├── runtime.py          # Task execution engine (TaskRunner, TaskFlowBuilder)
├── mock.py             # In-memory mock backend for testing
├── basis/
│   └── mujoco_env.py   # UnifiedMujocoEnv — Mujoco wrapper with sensor suite
├── backend/
│   └── mjc/            # Mujoco backend (operators, objects, scene)
└── utils/
    └── pose.py         # Pose transforms and quaternion utilities
```

**Execution flow:**

1. Load and validate the YAML task file via Pydantic models
2. Instantiate the backend via the configured factory function
3. `TaskFlowBuilder` expands stages into primitive pose-move and EEF-control actions
4. `TaskRunner.reset()` initializes the scene (with optional randomization)
5. `TaskRunner.update()` advances one step of the active primitive action
6. After completion, `TaskRunner.records` holds per-stage execution history

## Implementing a Custom Backend

To integrate a new simulator or real robot, implement three abstract classes from `auto_atom.runtime`:

- `OperatorHandler` — arm movement (`move_to_pose`) and gripper control (`control_eef`)
- `ObjectHandler` — object pose queries and updates (`get_pose`, `set_pose`)
- `SceneBackend` — scene lifecycle, handler resolution, and randomization

See the **[Custom Backend Guide](docs/custom-backend.md)** for a step-by-step walkthrough with annotated code examples. [`auto_atom/mock.py`](auto_atom/mock.py) provides a minimal reference implementation.

## License

See [LICENSE](LICENSE).
