# Implementing a Custom Backend

A backend connects the task runner to a specific robot or simulator. The runtime is batch-first: every public observation/action/state interface should carry a leading `env_dim`, and single-env execution is represented as `batch_size=1`. The recommended approach follows a two-layer pattern:

1. **Basis environment** (`auto_atom/basis/`) — a standalone class that wraps the simulator or hardware SDK, with no dependency on the task runner framework.
2. **Backend adapters** (`auto_atom/backend/<name>/`) — thin wrappers that implement the framework interfaces (`OperatorHandler`, `ObjectHandler`, `SceneBackend`) on top of the basis environment.

This separation keeps the basis environment reusable for other purposes (e.g. data collection, RL training) without pulling in task runner logic.

The built-in Mujoco integration in [`auto_atom/basis/mjc/mujoco_env.py`](../auto_atom/basis/mjc/mujoco_env.py) and [`auto_atom/backend/mjc/`](../auto_atom/backend/mjc/) follows exactly this pattern and serves as the canonical reference.

---

## YAML configuration and Hydra

Task files are loaded via [Hydra](https://hydra.cc). Every value in the YAML file is processed by Hydra's `instantiate` / `OmegaConf`, so the full Hydra feature set is available:

- **`_target_`** — instantiate any Python class or call any callable
- **Variable interpolation** — `${key}` references within the same file
- **Overrides** — pass `key=value` on the command line to override any field
- **Structured configs** — nest configs freely; lists and dicts are preserved as-is

A typical task file structure:

```yaml
# --- Environment (Hydra instantiation) ---
env:
  _target_: auto_atom.runtime.ComponentRegistry.register_env
  name: my_env
  env:
    _target_: my_package.basis.my_env.MyEnv
    scene_path: path/to/scene.xml
    headless: true
    render_width: 1280
    render_height: 720

# --- Backend factory (dotted import path) ---
backend: my_package.backend.my_backend.build_my_backend

# --- Task definition ---
task:
  env_name: my_env          # must match the name registered above
  seed: 42
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

# --- Operator declarations ---
task_operators:
  arm_a: {}
```

> [!NOTE]
> The `env` section is evaluated by Hydra before the backend factory is called. `ComponentRegistry.register_env` stores the constructed batched environment object under the given `name` so the factory can retrieve it with `ComponentRegistry.get_env(config.env_name)`.

---

## Step 1 — Implement the basis environment

Create a self-contained environment class under `auto_atom/basis/`. It should only depend on the simulator SDK and general utilities — **not** on `auto_atom.runtime` or `auto_atom.framework`.

```
auto_atom/
└── basis/
    └── my_env.py     ← new file
```

```python
# auto_atom/basis/my_env.py
"""Standalone environment wrapper for MySim — no framework dependency."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MyEnv:
    """Wraps the MySim SDK and exposes a clean, framework-agnostic interface."""

    scene_path: str
    headless: bool = True
    render_width: int = 640
    render_height: int = 480

    # Internal SDK handle, populated by setup().
    _sim: Any = field(default=None, init=False, repr=False)

    def setup(self) -> None:
        """Load the scene and initialise the simulator."""
        import my_sim_sdk
        self._sim = my_sim_sdk.load(self.scene_path, headless=self.headless)

    def reset(self) -> None:
        """Return the simulation to its initial state."""
        self._sim.reset()

    def step(self) -> None:
        """Advance the simulation by one timestep."""
        self._sim.step()

    def close(self) -> None:
        self._sim.close()

    # --- Pose queries (world frame, XYZW quaternion) ---

    def get_body_pose(self, name: str) -> tuple:
        """Return (position, quaternion) for a named body."""
        return self._sim.get_body_position(name), self._sim.get_body_quat(name)

    def set_body_pose(self, name: str, position, quaternion) -> None:
        self._sim.teleport_body(name, position, quaternion)

    def get_eef_pose(self, arm_id: str) -> tuple:
        return self._sim.get_eef_position(arm_id), self._sim.get_eef_quat(arm_id)

    def get_base_pose(self, arm_id: str) -> tuple:
        return self._sim.get_base_position(arm_id), self._sim.get_base_quat(arm_id)

    # --- Control ---

    def step_arm_towards(self, arm_id: str, position, orientation) -> bool:
        """Advance the arm by one step. Returns True when the goal is reached."""
        return self._sim.step_towards(arm_id, position, orientation)

    def step_gripper(self, arm_id: str, close: bool) -> bool:
        """Advance the gripper by one step. Returns True when done."""
        return self._sim.close_gripper(arm_id) if close else self._sim.open_gripper(arm_id)

    # --- Grasp state ---

    def is_grasped(self, arm_id: str, object_name: str) -> bool:
        return self._sim.check_grasp(arm_id, object_name)

    def is_grasping(self, arm_id: str) -> bool:
        return self._sim.check_any_grasp(arm_id)
```

---

## Step 2 — Implement `ObjectHandler`

Create the backend adapters under `auto_atom/backend/<name>/`. Each adapter holds a reference to the shared basis environment instance.

```
auto_atom/
└── backend/
    └── my_backend/
        ├── __init__.py
        └── my_backend.py   ← new file
```

```python
# auto_atom/backend/my_backend/my_backend.py
from dataclasses import dataclass
from auto_atom.runtime import ObjectHandler
from auto_atom.utils.pose import PoseState
from auto_atom.basis.my_env import MyEnv


@dataclass
class MyObjectHandler(ObjectHandler):
    # `name` (str) is inherited from ObjectHandler — do not redeclare it.
    env: MyEnv
    body_name: str  # may differ from the task-level object name

    def get_pose(self) -> PoseState:
        position, quaternion = self.env.get_body_pose(self.body_name)
        return PoseState(position=tuple(position), orientation=tuple(quaternion))

    def set_pose(self, pose: PoseState) -> None:
        self.env.set_body_pose(self.body_name, pose.position, pose.orientation)
```

`PoseState` convention:

| Field | Type | Convention |
|-------|------|-----------|
| `position` | `tuple[float, float, float]` | metres, XYZ |
| `orientation` | `tuple[float, float, float, float]` | quaternion, XYZW |

---

## Step 3 — Implement `OperatorHandler`

The operator is called once per control loop tick. Each method advances the action by one step and returns a `ControlResult`.

```python
from auto_atom.runtime import ControlResult, ControlSignal, ObjectHandler, OperatorHandler
from auto_atom.framework import EefControlConfig, PoseControlConfig
from auto_atom.utils.pose import PoseState
from auto_atom.basis.my_env import MyEnv


class MyOperatorHandler(OperatorHandler):

    def __init__(self, arm_id: str, env: MyEnv):
        self._arm_id = arm_id
        self._env = env

    @property
    def name(self) -> str:
        return self._arm_id

    def move_to_pose(
        self,
        pose: PoseControlConfig,
        target: ObjectHandler | None,
    ) -> ControlResult:
        """Advance motion toward `pose` by one control tick.

        The runtime resolves the target pose into the world frame before
        calling this method, so `pose.position` and `pose.orientation`
        are already in world-frame coordinates.
        Return REACHED once the arm arrives; RUNNING on every earlier tick.
        """
        done = self._env.step_arm_towards(
            self._arm_id,
            position=pose.position,
            orientation=pose.orientation,
        )
        return ControlResult(signal=ControlSignal.REACHED if done else ControlSignal.RUNNING)

    def control_eef(self, eef: EefControlConfig) -> ControlResult:
        """Advance the gripper toward the desired state by one control tick."""
        done = self._env.step_gripper(self._arm_id, close=eef.close)
        return ControlResult(signal=ControlSignal.REACHED if done else ControlSignal.RUNNING)

    def get_end_effector_pose(self) -> PoseState:
        pos, quat = self._env.get_eef_pose(self._arm_id)
        return PoseState(position=tuple(pos), orientation=tuple(quat))

    def get_base_pose(self) -> PoseState:
        pos, quat = self._env.get_base_pose(self._arm_id)
        return PoseState(position=tuple(pos), orientation=tuple(quat))
```

### `ControlSignal` values

| Signal | Meaning |
|--------|---------|
| `RUNNING` | Primitive action is in progress — call again next tick |
| `REACHED` | Primitive action completed successfully |
| `TIMED_OUT` | Action exceeded a time limit — runner marks stage as failed |
| `FAILED` | Unrecoverable error — runner marks stage as failed |

---

## Step 4 — Implement `SceneBackend`

`SceneBackend` owns the scene lifecycle, resolves handler instances by name, and reports grasp state.

```python
from auto_atom.runtime import SceneBackend, OperatorHandler, ObjectHandler
from auto_atom.framework import AutoAtomConfig
from auto_atom.basis.my_env import MyEnv


class MySceneBackend(SceneBackend):

    def __init__(
        self,
        env: MyEnv,
        operators: dict[str, MyOperatorHandler],
        objects: dict[str, MyObjectHandler],
    ):
        self._env = env
        self._operators = operators
        self._objects = objects

    # --- Lifecycle ---

    def setup(self, config: AutoAtomConfig) -> None:
        """Called once before the first reset. Initialise the basis environment."""
        self._env.setup()

    def reset(self) -> None:
        """Called before each task run. Return the scene to its initial state."""
        self._env.reset()

    def teardown(self) -> None:
        """Called when TaskRunner.close() is invoked."""
        self._env.close()

    # --- Handler resolution ---

    def get_operator_handler(self, name: str) -> OperatorHandler:
        return self._operators[name]

    def get_object_handler(self, name: str) -> ObjectHandler | None:
        if not name:
            return None
        return self._objects[name]

    # --- Grasp state ---

    def is_object_grasped(self, operator_name: str, object_name: str) -> bool:
        return self._env.is_grasped(operator_name, object_name)

    def is_operator_grasping(self, operator_name: str) -> bool:
        return self._env.is_grasping(operator_name)

    # --- Optional: task-focus notification ---

    def set_interest_objects_and_operations(
        self,
        object_names: list[str],
        operation_names: list[str],
    ) -> None:
        """Called by the runner before each stage. Use it to focus sensors, rendering, etc."""
        pass
```

---

## Step 5 — Write the factory function

The factory is called by the framework after Hydra has instantiated and registered the environment. It must accept the task config and operator list and return a `SceneBackend`.

```python
from typing import Any
from auto_atom.runtime import ComponentRegistry
from auto_atom.framework import AutoAtomConfig, OperatorConfig
from auto_atom.basis.my_env import MyEnv


def build_my_backend(
    task: AutoAtomConfig | dict[str, Any],
    operators: dict[str, OperatorConfig],
) -> MySceneBackend:
    # TaskFileConfig already validated `operators` (each value is an
    # OperatorConfig with its `name` populated from the dict key).
    config = (
        task if isinstance(task, AutoAtomConfig)
        else AutoAtomConfig.model_validate(task)
    )
    operator_configs = list(operators.values())

    # Retrieve the basis environment registered by the `env` YAML section.
    env: MyEnv = ComponentRegistry.get_env(config.env_name)

    # Build operator handlers.
    operator_handlers = {
        op.name: MyOperatorHandler(arm_id=op.name, env=env)
        for op in operator_configs
    }

    # Build object handlers for every object referenced in the task.
    object_names = {stage.object for stage in config.stages if stage.object}
    object_handlers = {
        name: MyObjectHandler(name=name, env=env, body_name=name)
        for name in object_names
    }

    return MySceneBackend(
        env=env,
        operators=operator_handlers,
        objects=object_handlers,
    )
```

---

## Step 6 — Write the YAML task file

Reference the factory and configure the environment with full Hydra support:

```yaml
# --- Hydra instantiates MyEnv and registers it under "my_env" ---
env:
  _target_: auto_atom.runtime.ComponentRegistry.register_env
  name: my_env
  env:
    _target_: my_package.basis.my_env.MyEnv
    scene_path: path/to/scene.xml
    headless: true
    render_width: 1280
    render_height: 720

# --- Backend factory ---
backend: my_package.backend.my_backend.build_my_backend

# --- Task ---
task:
  env_name: my_env
  seed: 42
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

task_operators:
  arm_a: {}

```

Because the file is processed by Hydra you can use all standard Hydra features. For example, to override fields at runtime:

```bash
python run_task.py scene_path=other_scene.xml task.seed=0
```

---

## Resulting file layout

After following this guide, the relevant files should look like:

```
auto_atom/
├── basis/
│   └── my_env.py              # standalone basis environment
└── backend/
    └── my_backend/
        ├── __init__.py
        └── my_backend.py      # ObjectHandler, OperatorHandler, SceneBackend, factory
```

---

## Reference implementation

The built-in Mujoco backend follows exactly this pattern:

| Layer | File |
|-------|------|
| Basis environment | [`auto_atom/basis/mjc/mujoco_env.py`](../auto_atom/basis/mjc/mujoco_env.py) |
| Backend adapters + factory | [`auto_atom/backend/mjc/mujoco_backend.py`](../auto_atom/backend/mjc/mujoco_backend.py) |
| Minimal mock (no external deps) | [`auto_atom/mock.py`](../auto_atom/mock.py) |
