# View Scene

Launch the interactive MuJoCo viewer on a composed scene + robot, applying every YAML-defined home-pose and pose override before the first frame is shown.

Since scene XMLs no longer have to embed a robot include or `<key name="home">`, opening a host XML directly can show only the host. This script reads `env.scene`, compiles its ordered MJCF and asset-assembly layers, applies `env.initial_joint_positions` plus `task.initial_pose` and `task_operators.<name>.initial_state.base_pose`, and hands the model to `mujoco.viewer.launch`.

**Script:** [examples/view_scene.py](../../examples/view_scene.py)

## Usage

```bash
python examples/view_scene.py --config-name pick_and_place
python examples/view_scene.py --config-name open_door_airbot_play_back_gs
python examples/view_scene.py --config-name open_door_p7_ik
python examples/view_scene.py --debug --config-name open_door_p7_ik
```

The default config is `pick_and_place`. Any Hydra override can be appended after `--`:

```bash
python examples/view_scene.py --config-name open_door_p7_ik -- env.initial_joint_positions.joint1=0.5
```

## What it composes

For the chosen config, the script reads these override surfaces and applies them on top of the host scene:

| Source                                            | What it sets                                                                  |
|---------------------------------------------------|-------------------------------------------------------------------------------|
| `env.scene.base`                                  | Host scene XML                                                                 |
| `env.scene.layers`                                | Ordered MJCF and namespaced asset-assembly layers                              |
| `env.sim_freq`                                   | Physics frequency, overriding the host XML timestep exactly as the runtime does |
| `env.initial_joint_positions`                     | Per-joint home pose (mirrors `MujocoBasis.reset()`)                           |
| `task.initial_pose`                               | Per-body pose overrides (freejoint qpos for movable bodies, or `body_pos/quat` for static bodies) |
| `task_operators.<name>.initial_state.base_pose`   | Relocates each operator's `root_body` so the arm sits at the right world pose |

Equality-constrained passive joints (e.g. parallel-linkage gripper followers) are settled by stepping under zero gravity while pinning the configured scalar joints, matching the runtime backend reset.

Configured joint-position actuators are initialized to hold the applied home
pose. Motor, velocity, tendon, and site actuators keep their existing controls.

Mocap bodies welded to a freejoint are synced onto their target pose so the arm doesn't snap on the first viewer step.

## Gaussian Splatting mode

When the chosen config carries an `env.gaussian_render` section with at least one body
gaussian or a background PLY (e.g. `open_door_airbot_play_gs`,
`stack_color_blocks_gs`), the script switches to a **passive** MuJoCo viewer
and opens a second OpenCV window titled
`GS view (synced with MuJoCo viewer)`.

The GS window re-renders the scene from the same free-camera pose as the
MuJoCo viewer every step, so orbit / pan / zoom in the MuJoCo viewer drives
the GS preview live. The window defaults to 640×480; use the OpenCV window
controls (or close the MuJoCo viewer) to exit.

Detection is content-based: the script looks for `env.gaussian_render` with
either `body_gaussians` or `background_ply` populated, so it works whether
the task uses the GS env target directly or composes GS into a non-GS env.

Multi-background configs (`background_ply` is list / glob / parts dict) are
previewed with **only the first resolved PLY** — the viewer is for verifying
geometry alignment, not for sweeping backgrounds.

## Reload workflow

Pick up edits without restarting Python:

- **GS mode** — press `R` in either window, or click the **Reload (R)** button
  in the top-right of the GS window. Both trigger a full reload.
- **Non-GS mode** — use the reload button on the MuJoCo viewer panel.

Reload re-reads:

- **YAML edits** to `env.initial_joint_positions`, `task.initial_pose`,
  `task_operators.<name>.initial_state.base_pose`, `env.scene.*`, or
  `env.gaussian_render.*` — re-composed via Hydra from disk.
- **XML/package edits** to the host, MJCF layer or asset package — re-read by
  `auto_atom.scene_composition.load_composed_scene`.
- **PLY edits** in GS mode — body / background gaussians are reloaded from
  disk and a new `GSRendererMuJoCo` is built. The GS window shows
  `Loading Gaussian renderer...` and `Warming up GS render...` status frames
  during the rebuild; the first GS frame after a reload can take a few
  seconds while gaussian PLYs upload to GPU.

In GS mode the MuJoCo viewer window is closed and reopened around each
reload (because the underlying `MjModel` is replaced); this is normal.

This makes `view_scene.py` the fastest way to iterate on home pose, scene
composition, geometry, and Gaussian alignment side by side.

## Console output

On startup and on every reload the script prints:

```
[info] scene  : .../scenes/open_door/demo.xml
[info] robots : ['.../robots/p7_arm_with_xf9600.xml']
[info] home   : 9 joint override(s), 0 body pose(s), 1 operator base(s)
[info] model  : nq=23 nv=22 nu=8 nbody=14 ngeom=37  (robots=[...], ijp=9, body_pose=0, op_base=1)
```

Use these counters to confirm that the expected robot was injected and that all overrides were honoured.

Pass `--debug` before Hydra arguments to preflight-build the model and print
full loader tracebacks to the terminal. This is useful for errors that MuJoCo's
viewer would otherwise show only inside the GUI.

## Related

- [Scene Composition](../task-configuration/scene_composition.md) — the shared layer compiler and package contract
- [Tune Randomization Extremes](tune_randomization_extremes.md) — same override surfaces, with randomization stepping
