# View Scene

Launch the interactive MuJoCo viewer on a composed scene + robot through the
canonical `MujocoTaskBackend` interface. The viewer is a geometry/placement
diagnostic; it does not own simulation state or duplicate task reset semantics.

Since scene XMLs no longer have to embed a robot include or `<key name="home">`, opening a host XML directly can show only the host. This script composes the
task file, constructs a `MujocoTaskBackend`, and calls its public
`setup()`/`reset()` lifecycle before handing the backend-owned model and
data to the viewer.  Initial joints, object/base poses, randomization, mocap
sync, and settle behavior therefore have one implementation in the backend,
shared with ``aao-demo`` and policy evaluation.

The application order is owned by the simulator backend:

1. construct the backend from the composed task file;
2. call `backend.setup()` once to establish the configured home state;
3. call `backend.reset()` so the viewer starts from the same reset state as
   the runtime task loop;
4. pass the selected backend environment to the native or GS viewer.

`view_scene.py` intentionally contains no second pose resolver, joint-home
application path, or simulator session wrapper.  It is a thin visualization
client over the backend seam.

Named pose references and reset ordering follow the backend contract.  The
viewer does not reinterpret them or create a second source of truth.

**Script:** [examples/view_scene.py](../../examples/view_scene.py)

## Usage

```bash
python examples/view_scene.py --config-name pick_and_place
python examples/view_scene.py --config-name open_door_airbot_play_back_gs
python examples/view_scene.py --config-name open_door_p7_ik
python examples/view_scene.py --debug --config-name open_door_p7_ik
python examples/view_scene.py --show-object-frames --config-name open_door_p7_ik
```

Press `Ctrl+C` in the terminal to close the viewer and tear down the backend;
no GUI click is required.

Pass `--show-object-frames` to start the MuJoCo window with **Frame → Body**
already enabled. MuJoCo displays axes for all model bodies, including task
objects and robot links. In GS mode this affects the synchronized MuJoCo window;
the Gaussian raster window itself does not render MuJoCo debug overlays.

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
| `env.initial_joint_positions`                     | Per-joint home pose consumed by the backend environment reset |
| `task.initial_pose`                               | Per-body `PoseOverrideConfig` consumed by `MujocoTaskBackend.setup/reset` |
| `task_operators.<name>.initial_state.base_pose`   | Operator base `PoseOverrideConfig` consumed by the backend, including named-frame and per-axis reference resolution |

Equality-constrained passive joints, configured joint-position actuators, and
mocap synchronization are all handled by the backend's normal setup/reset
implementation, exactly as they are for runtime execution.

`task_operators.<name>.initial_state.eef_pose`, the `eef` gripper scalar, and
camera initial poses are consumed by the backend setup/reset contract as
supported by the selected backend. The standalone viewer does not duplicate
those operations or reinterpret their references; use `aao-demo` when you
need to validate task execution after the static scene inspection.

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
  `task_operators.<name>.initial_state.base_pose` (and warnings for
  `initial_state.eef_pose`), `task.camera_initial_pose`,
  `env.scene.*`, or
  `env.gaussian_render.*` — re-composed via Hydra from disk.
- **XML/package edits** to the host, MJCF layer or asset package — re-read by
  `auto_atom.scene_composition.load_composed_scene`.
- **PLY edits** in GS mode — body / background gaussians are reloaded from
  disk and a new `GSRendererMuJoCo` is built. The GS window shows
  `Loading Gaussian renderer...` and `Warming up GS render...` status frames
  during the rebuild; the first GS frame after a reload can take a few
  seconds while gaussian PLYs upload to GPU.

In GS mode the MuJoCo viewer window is closed and reopened around each
reload (because a replacement backend owns a new `MjModel`); this is normal.
The old backend is torn down only after the replacement has been successfully
constructed, and failed replacement loads are torn down immediately.

This makes `view_scene.py` the fastest way to iterate on home pose, scene
composition, geometry, and Gaussian alignment side by side.

## Console output

On startup and after a successful reload the script prints the compiled model
dimensions, for example:

```
[info] model  : nq=23 nv=22 nu=8 nbody=14 ngeom=37
```

Pass `--debug` before Hydra arguments to print full tracebacks for Gaussian
render and reload errors.  Backend construction/setup failures are propagated
normally, so the original exception and its traceback remain visible.

## Related

- [Scene Composition](../task-configuration/scene_composition.md) — the shared layer compiler and package contract
- [Tune Randomization Extremes](tune_randomization_extremes.md) — same override surfaces, with randomization stepping
