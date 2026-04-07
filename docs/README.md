# Auto Atomic Operation

[![PyPI](https://img.shields.io/pypi/v/auto-atomic-operation)](https://pypi.org/project/auto-atomic-operation/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/OpenGHz/auto-atomic-operation/blob/main/LICENSE)

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

# Core framework only
pip install -e .

# With the built-in MuJoCo backend
pip install -e ".[mujoco]"
```

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

### 3. Run with CLI

```bash
# Mock backend (no simulator required)
aao_demo --config-name mock

# MuJoCo demos
aao_demo --config-name pick_and_place

# List all available demos
aao_demo --list
```

## MuJoCo Demos

| | |
|:---:|:---:|
| ![pick_and_place](assets/videos/pick_and_place.gif) | ![cup_on_coaster](assets/videos/cup_on_coaster.gif) |
| `pick_and_place` | `cup_on_coaster` |
| ![stack_color_blocks](assets/videos/stack_color_blocks.gif) | ![press_three_buttons](assets/videos/press_three_buttons.gif) |
| `stack_color_blocks` | `press_three_buttons` |

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

## License

[MIT](https://github.com/OpenGHz/auto-atomic-operation/blob/main/LICENSE)
