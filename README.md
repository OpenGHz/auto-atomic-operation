<div align="center">

<img src="assets/logo.svg" alt="Auto Atomic Operation Logo" width="200">

<h1>Auto Atomic Operation</h1>

[![PyPI](https://img.shields.io/pypi/v/auto-atomic-operation)](https://pypi.org/project/auto-atomic-operation/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=readthedocs)](https://openghz.github.io/auto-atomic-operation/#/)

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
- **3D Gaussian Splatting rendering** — photorealistic rendering for any task via `_gs.yaml` configs

## Installation

Requires **Python 3.10+**.

### Install from PyPI

```bash
pip install auto-atomic-operation
```

The PyPI release lags behind source and does not ship with demo configs/assets. For the latest features and runnable demos, install from source.

### Install from source

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/OpenGHz/auto-atomic-operation.git
cd auto-atomic-operation

pip install -e .              # core framework only
pip install -e ".[mujoco]"    # with the built-in MuJoCo backend
pip install -e ".[gs]"        # with 3D Gaussian Splatting rendering
```

MuJoCo demo assets live in Git LFS. After cloning, install Git LFS and pull them:

```bash
sudo apt-get install git-lfs   # Debian/Ubuntu
git lfs pull
```

## Running demos

The CLI looks for configs in `./aao_configs/` relative to the current working directory. After cloning the repo, run from the project root:

```bash
aao-info                                 # list every runnable task
aao-demo --config-name mock              # mock backend, no simulator required
aao-demo --config-name pick_and_place    # MuJoCo demo (default)
aao-demo --config-name <config>          # any other config
```

For all CLI flags, Hydra overrides, and `aao-eval` usage, see the [CLI Reference](https://openghz.github.io/auto-atomic-operation/#/getting-started/cli_reference).

### Robotiq tasks

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

### Franka task

| |
|:---:|
| ![pick_and_place_franka](assets/videos/pick_and_place_franka.gif) |
| `pick_and_place_franka` |

### 3D Gaussian Splatting demos

GS configs are named `<task>_gs` and run identically to the native MuJoCo demos. Asset bundles live on Hugging Face:

```bash
pip install huggingface_hub "httpx[socks]"
hf download OpenGHz/auto-atom-assets --repo-type=dataset --include "assets/gs/*" --local-dir .
```

| | |
|:---:|:---:|
| ![cup_on_coaster_gs](assets/videos/cup_on_coaster_gs.gif) | ![stack_color_blocks_gs](assets/videos/stack_color_blocks_gs.gif) |
| `cup_on_coaster_gs` | `stack_color_blocks_gs` |
| ![press_three_buttons_gs](assets/videos/press_three_buttons_gs.gif) | ![arrange_flowers_gs](assets/videos/arrange_flowers_gs.gif) |
| `press_three_buttons_gs` | `arrange_flowers_gs` |
| ![hang_toothbrush_cup_gs](assets/videos/hang_toothbrush_cup_gs.gif) | ![wipe_the_table_gs](assets/videos/wipe_the_table_gs.gif) |
| `hang_toothbrush_cup_gs` | `wipe_the_table_gs` |

## Documentation

Full documentation is hosted at **[openghz.github.io/auto-atomic-operation](https://openghz.github.io/auto-atomic-operation/#/)**. Common entry points:

- [CLI Reference](https://openghz.github.io/auto-atomic-operation/#/getting-started/cli_reference) — every flag, override, and output of `aao-demo` / `aao-eval`
- [Task File Schema](https://openghz.github.io/auto-atomic-operation/#/task-configuration/task_file_schema) — top-level YAML keys, stages, pose references, and execution policy
- [Task Configuration](https://openghz.github.io/auto-atomic-operation/#/task-configuration/stages_and_waypoints) — waypoints, interval boundaries, stage sites, randomization, and scene composition
- [Execution Completion Flow](https://openghz.github.io/auto-atomic-operation/#/task-configuration/execution_completion_flow) — how `pre_move` / `eef` / `post_move` decide they are done and how that drives stage success
- [MuJoCo Backend](https://openghz.github.io/auto-atomic-operation/#/mujoco-backend/mujoco_backend_conditions) — backend conditions, gripper semantics, observation wiring
- [Custom Backend Guide](https://openghz.github.io/auto-atomic-operation/#/mujoco-backend/custom-backend) — integrate a new simulator or real robot
- [IK & Motion Control](https://openghz.github.io/auto-atomic-operation/#/ik-motion-control/ik_control)
- [Gaussian Splatting](https://openghz.github.io/auto-atomic-operation/#/gaussian-splatting/gs_rendering_alignment)
- [Tools](https://openghz.github.io/auto-atomic-operation/#/tools/data_collection) — data collection, replay, policy evaluation, panel XML assembly, scene viewer, randomization tuning, benchmarking
- [Migration Notes](https://openghz.github.io/auto-atomic-operation/#/migration-notes/xml_mesh_gs_migration_notes) — bring your own XML / mesh / GS assets into this project's layout

## License

See [LICENSE](LICENSE).
