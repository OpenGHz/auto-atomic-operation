# Auto Atomic Operation

[![PyPI](https://img.shields.io/pypi/v/auto-atomic-operation)](https://pypi.org/project/auto-atomic-operation/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/OpenGHz/auto-atomic-operation/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-OpenGHz%2Fauto--atomic--operation-181717?logo=github)](https://github.com/OpenGHz/auto-atomic-operation)

> A YAML-driven atomic operation framework for robotic manipulation.

`auto-atomic-operation` lets you define robotic manipulation tasks — move, grasp, release, pick, place, push, pull, press — as declarative YAML files. A built-in state machine handles task sequencing, pose resolution, end-effector control, and execution tracking. A plugin-based backend system decouples task logic from the underlying hardware or simulator, making it easy to run the same task definition against a real robot, a physics simulator, or a lightweight mock for testing.

## Features

- **Hydra-powered task configuration** — describe multi-stage manipulation tasks in YAML; full [Hydra](https://hydra.cc) support means `_target_` instantiation, variable interpolation, and command-line overrides work out of the box
- **Flexible pose references** — specify targets relative to world, robot base, end-effector, or tracked objects
- **Pluggable backends** — connect any robot or simulator by implementing a small set of abstract interfaces
- **Pose randomization** — per-object position/orientation randomization with automatic collision avoidance on reset
- **Multi-arm support** — single-arm and dual-arm (left/right) topologies
- **Execution records** — detailed per-stage status, failure reasons, and timing after every run
- **MuJoCo backend included** — a ready-to-use backend with RGB-D cameras, tactile sensors, force/torque, IMU, and joint state support
- **3D Gaussian Splatting rendering** — photorealistic task rendering through composable GS configs

## Installation

Requires **Python 3.10+**.

### Install from PyPI

```bash
pip install auto-atomic-operation
```

### Install from source

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/OpenGHz/auto-atomic-operation.git
cd auto-atomic-operation

pip install -e .              # core framework only
pip install -e ".[mujoco]"   # with the built-in MuJoCo backend
pip install -e ".[gs]"       # with 3D Gaussian Splatting rendering
```

The PyPI release can lag behind source and does not ship the repository's demo
configs or assets.  MuJoCo meshes and videos live in Git LFS:

```bash
sudo apt-get install git-lfs   # Debian/Ubuntu
git lfs pull
```

Gaussian scene assets are hosted separately on Hugging Face:

```bash
pip install huggingface_hub "httpx[socks]"
hf download OpenGHz/auto-atom-assets --repo-type=dataset --include "assets/gs/*" --local-dir .
```

## Quick Start

### 1. Define a task in YAML

```yaml
# task.yaml
env:
  _target_: auto_atom.mock.create_mock_env
  name: my_env
  kind: mock_env

backend: auto_atom.mock.build_mock_backend

task:
  env_name: my_env
  stages:
    - name: approach_cup
      object: cup
      operation: move
      operator: arm_a
      param:
        pre_move:
          - position: [0.45, -0.10, 0.08]
            rotation: [0.0, 1.57, 0.0]
            reference: object_world

    - name: move_to_shelf
      object: shelf
      operation: move
      operator: arm_a
      param:
        pre_move:
          - position: [0.10, 0.25, 0.16]
            orientation: [0.0, 0.0, 0.0, 1.0]
            reference: world

task_operators:
  arm_a: {}
```

See [Task File Schema](task-configuration/task_file_schema.md) for the complete
top-level, stage, control, pose-reference, and execution-policy fields.

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

### 3. Run with CLI

```bash
# Mock backend (no simulator required)
aao-demo --config-name mock

# MuJoCo demos
aao-demo --config-name pick_and_place

# List all runnable tasks (name, objects, operations, workflow)
aao-info
```

Use `aao-info` rather than a static task list so composed tasks and newly added
robot variants are included.  To inspect the fully composed scene and robot,
use the [View Scene](tools/view_scene.md) tool instead of opening a host XML
directly.

## MuJoCo Demos

| | |
|:---:|:---:|
| ![pick_and_place](https://media.githubusercontent.com/media/OpenGHz/auto-atomic-operation/main/assets/videos/pick_and_place.gif) | ![cup_on_coaster](https://media.githubusercontent.com/media/OpenGHz/auto-atomic-operation/main/assets/videos/cup_on_coaster.gif) |
| `pick_and_place` | `cup_on_coaster` |
| ![stack_color_blocks](https://media.githubusercontent.com/media/OpenGHz/auto-atomic-operation/main/assets/videos/stack_color_blocks.gif) | ![press_three_buttons](https://media.githubusercontent.com/media/OpenGHz/auto-atomic-operation/main/assets/videos/press_three_buttons.gif) |
| `stack_color_blocks` | `press_three_buttons` |

## Architecture

```
auto_atom/
├── framework.py        # Pydantic configuration models (YAML schema)
├── runtime.py          # TaskRunner, primitive actions, and backend protocols
├── stage_execution.py  # Shared stage state machine and condition checks
├── execution_timeline.py # Stable stage/keypoint/primitive ordering
├── scene_composition/  # Host MJCF + ordered asset-layer compiler
├── policy_eval.py      # External-policy evaluator and shared result types
├── mock.py             # In-memory mock backend for testing
├── basis/
│   └── mjc/
│       └── mujoco_env.py   # UnifiedMujocoEnv — Mujoco wrapper with sensor suite
├── backend/
│   └── mjc/            # Mujoco backend (operators, objects, scene)
├── runner/              # aao-demo, aao-eval, and aao-info entry points
└── utils/
    └── pose.py         # Pose transforms and quaternion utilities
```

The execution path is intentionally layered: Hydra composes and validates a
task file, the backend creates handlers over a basis environment,
`TaskFlowBuilder` expands stages into primitives, and `StageExecution` applies
operation conditions while `TaskRunner` advances the simulation.  The detailed
phase and condition flow is documented in [Execution Completion Flow](task-configuration/execution_completion_flow.md);
the interface boundary for another simulator or robot is covered by
[Implementing a Custom Backend](mujoco-backend/custom-backend.md).

## License

[MIT](https://github.com/OpenGHz/auto-atomic-operation/blob/main/LICENSE)
