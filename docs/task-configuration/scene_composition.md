# Scene Composition

`auto_atom` composes the scene XML and the robot XML(s) at load time instead of authoring a separate per-robot demo file. Scene XMLs declare only task-specific geometry (tables, objects, cameras, sites); robot XMLs are pulled in via `env.robot_paths` and injected as `<include>` siblings directly under `<mujoco>`.

## Why

Before this refactor, each robot needed its own `demo_<robot>.xml` (e.g. `demo_p7_xf9600.xml`, `demo_franka.xml`) duplicating the entire scene plus that robot's keyframe. Adding a new robot meant copy-pasting and re-tuning every task scene.

Now:

- One `assets/xmls/scenes/<task>/demo.xml` holds the scene only — no robot include, no `<key>` keyframe.
- Robots live under `assets/xmls/robots/` and are referenced by config.
- Home pose is described once in YAML as `env.initial_joint_positions` instead of an XML keyframe.

## Configuring `robot_paths`

Set `env.robot_paths` to the list of robot XMLs the task should compose with the scene. Order matters — earlier entries are inserted first. Most basis configs already do this for you:

```yaml
# aao_configs/basis_p7_xf9600.yaml
env:
  robot_paths:
    - ${assets_dir}/xmls/robots/p7_arm_with_xf9600.xml
  initial_joint_positions:
    joint1: 0.0
    joint2: -0.785
    # ... 7 P7 hinges + XF9600 gripper joints
```

When the scene XML already embeds its own robot (legacy monolithic scenes), leave `robot_paths` empty (the default in [`aao_configs/basis.yaml`](../../aao_configs/basis.yaml)). The loader takes a fast path and skips XML rewriting.

## Loader behaviour

`auto_atom.utils.scene_loader.load_scene` is the single entry point:

| Case                          | Behaviour                                                                        |
|-------------------------------|----------------------------------------------------------------------------------|
| `robot_paths=[]`              | `mujoco.MjModel.from_xml_path(scene_xml)` — no rewriting                         |
| `robot_paths=[r1, r2, ...]`   | Parse `scene_xml`, prepend `<include file="<abs path>"/>` for each robot under `<mujoco>`, write the composed XML to a temporary sibling of the scene file (so `meshdir` / relative paths still resolve), load it, then delete the temp file |

Robot include paths are made absolute, but the composed XML is written **next to the scene file**, which keeps relative `meshdir`, `texturedir`, and `<include>` references inside the scene resolving the same way they did before.

## Home pose injection

Because the scene XML no longer carries a `<key>` keyframe, the runtime applies `env.initial_joint_positions` on every reset:

- Scalar joint → write directly into `data.qpos`.
- Multi-DOF joint (free / ball) → write the full `[x y z qw qx qy qz]` or `[qw qx qy qz]` vector.
- Equality-constrained passive joints (e.g. parallel-linkage gripper followers) are settled by stepping under zero gravity while pinning the configured scalar joints.

This is implemented in `MujocoBasis.reset()`. The same logic is mirrored in [`examples/view_scene.py`](../../examples/view_scene.py) so the viewer shows exactly what the runtime will see.

## Available robot XMLs

| Robot XML                                  | Description                                       |
|--------------------------------------------|---------------------------------------------------|
| `assets/xmls/robots/robotiq.xml`           | 6-DOF floating base + Robotiq 2F-85 gripper       |
| `assets/xmls/robots/panda_robotiq.xml`     | Franka Panda + Robotiq 2F-85                      |
| `assets/xmls/robots/p7_arm_with_xf9600.xml`| 7-DOF P7 arm + XFG-9600 parallel-linkage gripper  |
| `assets/xmls/robots/airbot_play.xml`       | Airbot Play 6-DOF arm                             |
| `assets/xmls/robots/airbot_play_with_xf9600.xml` | Airbot Play + XFG-9600                      |
| `assets/xmls/robots/xf9600_mocap.xml`      | Mocap-driven floating XFG-9600 gripper            |

## Iterating with the viewer

[`examples/view_scene.py`](../../examples/view_scene.py) (see [View Scene](../tools/view_scene.md)) is the fastest way to verify a `robot_paths` change: it composes the scene + robot, applies all home-pose / initial-pose / operator base overrides, and supports reload-on-edit so you don't need to restart Python after tweaking YAML or XML.

## Related

- [Action Space](action_space.md) — how operators map joints/sites to actions
- [View Scene](../tools/view_scene.md) — interactive viewer that mirrors the runtime composition
- [Custom Backend](../mujoco-backend/custom-backend.md) — backend factories that bind to the composed model
