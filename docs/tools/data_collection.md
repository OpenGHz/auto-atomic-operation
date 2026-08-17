# Data Collection

This guide covers the scripts used to record task demonstrations and compare rendering outputs.

## Record Demo Video

[`examples/record_demo.py`](../examples/record_demo.py) runs a task in the MuJoCo backend and saves the camera feed as a GIF and/or MP4. It also exports replayable demo data, including low-dimensional observations, action pose targets, and low-level control actions. It reuses the same Hydra configs as `aao-demo`.

### Basic usage

```bash
python examples/record_demo.py --config-name pick_and_place
```

### Hide the robot from native camera data

Set `env.hide_operators_in_camera=true` to remove every configured operator
(`root_body` and all descendants, including gripper collision and tactile
geoms) from native MuJoCo RGB, depth, mask, and heat-map rendering:

```bash
aao-demo --config-name pick_and_place \
    env.hide_operators_in_camera=true +perf_count=true

python examples/record_demo.py --config-name pick_and_place \
    env.hide_operators_in_camera=true
```

The robot still participates in physics, contact, tactile sensing, and control.
A wrist-mounted camera also continues to move normally because cameras are not
geoms. The passive viewer remains unchanged; filtering applies only to captured
offscreen camera scenes.

When AAO is launched through AIRDC, the task config is nested under the manager:

```bash
airdc --name aao_config managers/auto_atom/task=pick_and_place \
    managers.auto_atom.env.hide_operators_in_camera=true
```

This option filters native MuJoCo rendering. Gaussian Splatting RGB/depth uses
a separate renderer; remove robot entries from `gaussian_render.body_gaussians`
(and ensure the background PLY does not bake in the robot) for GS datasets.

Output files are written to:

- `assets/videos/<config_name>.gif`
- `assets/videos/<config_name>.mp4`
- `assets/demos/<config_name>.json`
- `assets/demos/<config_name>.npz`

When `env.batch_size > 1`, the recorder writes one video per env replica:

- `assets/videos/<config_name>_env0.gif`
- `assets/videos/<config_name>_env0.mp4`
- `assets/videos/<config_name>_env1_cam.gif`
- ...

### Recorder options

All options are injected via Hydra with the `+recorder.` prefix:

| Option | Default | Description |
| ------ | ------- | ----------- |
| `camera` | `env1_cam` | Camera name to capture from |
| `fps` | `25` | Frame rate for the recording |
| `gif_width` | `320` | Width (pixels) of the output GIF |
| `max_updates` | `null` | Maximum number of `runner.update()` calls before auto-stopping |
| `save_gif` | `false` | Whether to save a GIF |
| `save_mp4` | `false` | Whether to save an MP4 |
| `save_demo` | `true` | Whether to save replayable demo metadata and action arrays |

```bash
# Save MP4 only
python examples/record_demo.py --config-name pick_and_place \
    +recorder.save_mp4=true +recorder.save_gif=false

# Use a different camera and higher FPS
python examples/record_demo.py --config-name cup_on_coaster \
    +recorder.camera=env0_cam +recorder.fps=30

# Wider GIF output
python examples/record_demo.py --config-name stack_color_blocks \
    +recorder.gif_width=480

# Stop recording automatically after 200 updates
python examples/record_demo.py --config-name press_three_buttons \
    +recorder.max_updates=200
```

## Replay Recorded Demo

[`examples/replay_demo.py`](../examples/replay_demo.py) replays data recorded by `record_demo.py`. It supports two replay modes:

- `ctrl`: replay the saved low-level MuJoCo control actions from `assets/demos/<config_name>.npz`
- `pose`: replay the saved `action/<operator>/pose` targets from `assets/demos/<config_name>.json` to validate whether the recorded pose targets are themselves sufficient

### Basic usage

```bash
# Replay saved low-level ctrl actions
python examples/replay_demo.py --config-name pick_and_place

# Replay saved action pose targets
python examples/replay_demo.py --config-name pick_and_place +replay.mode=pose

# Replay another recorded demo name
python examples/replay_demo.py --config-name pick_and_place \
    +replay.demo_name=my_demo +replay.mode=pose
```

Replay videos are written to `outputs/records/videos/`. Batched replays write
one file per environment, and multi-camera captures write one file per
environment/camera pair, for example `<demo_name>_env0_env2_cam_replay.mp4`.

### Replay options

All options are injected via Hydra with the `+replay.` prefix:

| Option | Default | Description |
| ------ | ------- | ----------- |
| `demo_name` | `<config_name>` | Demo basename under `assets/demos/` |
| `mode` | `ctrl` | Replay mode, either `ctrl` or `pose` |
| `camera` | `env1_cam` | Camera name used for replay recording |
| `fps` | `25` | Frame rate for replay video export |
| `gif_width` | `320` | Width (pixels) of the replay GIF |
| `save_gif` | `true` | Whether to save a replay GIF |
| `save_mp4` | `false` | Whether to save a replay MP4 |

```bash
# Save replay MP4 only
python examples/replay_demo.py --config-name open_hinge_door \
    +replay.save_mp4=true +replay.save_gif=false

# Use pose replay to validate recorded pose targets
python examples/replay_demo.py --config-name open_hinge_door \
    +replay.mode=pose

# Replay from another camera
python examples/replay_demo.py --config-name cup_on_coaster \
    +replay.camera=env0_cam
```

## Compare GS Render

[`examples/compare_gs_render.py`](../examples/compare_gs_render.py) renders the initial frame of a GS-enabled scene from every configured camera in both Gaussian Splatting and native MuJoCo modes, then saves a side-by-side comparison image.

### Basic usage

```bash
# Default config: press_three_buttons_gs
python examples/compare_gs_render.py

# Specify a different GS scene config
python examples/compare_gs_render.py --config-name cup_on_coaster_gs
python examples/compare_gs_render.py --config-name stack_color_blocks_gs
python examples/compare_gs_render.py --config-name hang_toothbrush_cup_gs
python examples/compare_gs_render.py --config-name wipe_the_table_gs
python examples/compare_gs_render.py --config-name arrange_flowers_gs
```

Output images are saved to `outputs/compare_<config_name>_<timestamp>.png`.

### Options

| Option | Default | Description |
| ------ | ------- | ----------- |
| `show` | `false` | Display the comparison interactively with matplotlib |

```bash
python examples/compare_gs_render.py show=true
```

## Compare Depth Render

[`examples/compare_depth_render.py`](../examples/compare_depth_render.py) renders the initial frame of a GS-enabled scene from every configured camera and compares three depth variants side-by-side:

- native MuJoCo depth buffer
- GS foreground accumulated depth
- GS composited depth using the `press_da_button` rule

### Basic usage

```bash
# Default config: press_three_buttons_gs
python examples/compare_depth_render.py

# Specify a different GS scene config
python examples/compare_depth_render.py --config-name cup_on_coaster_gs
python examples/compare_depth_render.py --config-name stack_color_blocks_gs
```

Output images are saved to `outputs/compare_depth_<config_name>_<timestamp>.png`.

### Options

| Option | Default | Description |
| ------ | ------- | ----------- |
| `show` | `false` | Display the comparison interactively with matplotlib |

```bash
python examples/compare_depth_render.py show=true
```

## Depth Source Notes

When collecting RGB-D data, "depth" can come from two different rendering pipelines in this repository and its third-party references. They are not numerically equivalent.

### Native MuJoCo depth

The current default depth path in [`auto_atom/basis/mjc/mujoco_env.py`](../auto_atom/basis/mjc/mujoco_env.py) uses MuJoCo's renderer directly:

```python
renderer.enable_depth_rendering()
depth = np.asarray(renderer.render(), dtype=np.float32)
renderer.disable_depth_rendering()
```

This produces a standard rasterized visible-surface depth map. It does not depend on Gaussian opacity or alpha compositing.

### GS depth

If the environment is switched to the GS renderer path in [`auto_atom/basis/mjc/gs_mujoco_env.py`](../auto_atom/basis/mjc/gs_mujoco_env.py), depth is generated by the Gaussian renderer instead. That file documents the GS depth mode as accumulated depth `∑w_i z_i`.

The same general behavior appears in the third-party `press_da_button` dataset pipeline. Its `justfile` launches dataset scripts that eventually call `self.env.batch_render(data)` and save the returned `depth_t`. In the underlying renderer implementation, depth is composited together with alpha:

```python
depth = fg_depth * alpha + bg_depth * (1 - alpha)
```

This means GS depth is:

- visually aligned with GS RGB rendering
- softer than a hard z-buffer at boundaries
- sensitive to Gaussian opacity / alpha quality
- more likely to contain blended depths near occlusion edges

If GS masks are generated with the same `alpha + scene_depth` logic as the third-party `press_one_button` pipeline, they should be interpreted as **visible-region masks**, not full object masks. Pixels belonging to object parts hidden behind other geometry are removed by occlusion culling. As a result, GS masks usually track what is visible in RGB, but can be softer or less stable near boundaries when opacity / depth quality is imperfect.

### Recommendation

- Use native MuJoCo depth when you need crisp visible-surface geometry or more conventional depth supervision.
- Use GS depth when RGB is also produced by GS and appearance consistency matters more than hard geometric depth.
- If a GS-generated point cloud looks noisy or "floating", inspect opacity / alpha behavior in addition to camera intrinsics and extrinsics.


## Integrate with an External Collector

To use AAO with another dataset writer, episode scheduler, or collection
framework, embed the runner and observation APIs directly. See
[Integrating AAO with External Data Collection Programs](external_data_collection.md)
for the framework-independent lifecycle, batch handling, schema adaptation,
and completion semantics.

## Robot Learning Data Collection

The AIRBOT-Data-Collection project integrates AAO and supports AAO-based data
collection. See its [AAO data collection workflow](https://github.com/OpenGHz/AIRBOT-Data-Collection/blob/develop/docs/workflows/aao.md)
for usage instructions.

### 模型训练

参考：https://github.com/OpenGHz/MCAP-DataLoader#integration-with-lerobot-training
