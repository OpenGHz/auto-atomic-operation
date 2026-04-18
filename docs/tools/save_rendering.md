# Save Rendering

Renders per-camera outputs from a task scene and saves them to disk. Supports both native MuJoCo and Gaussian Splatting backends.

**Script:** [examples/save_rendering.py](../examples/save_rendering.py)

## Usage

```bash
python examples/save_rendering.py
python examples/save_rendering.py --config-name press_three_buttons
python examples/save_rendering.py --config-name press_three_buttons_gs
python examples/save_rendering.py show=true
```

The default config is `press_three_buttons`. Use any GS config (e.g. `press_three_buttons_gs`) to render with Gaussian Splatting.

Example for the open-door back-side GS config and the test override:

```bash
/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
  examples/save_rendering.py \
  --config-name open_door_airbot_play_back_gs \
  +test=open_the_door
```

## Saved outputs

For each configured camera, the script saves:

| File | Content |
|---|---|
| `<camera>_rgb.png` | RGB color image |
| `<camera>_mask.png` | Binary segmentation mask (when available) |
| `<camera>_overlay.png` | RGB with mask overlay in red (when mask is available) |
| `<camera>_depth.png` | Colorized depth map using the turbo colormap (when available) |
| `<camera>_heatmap.png` | Multi-channel operation heatmap with color legend (when available) |
| `<camera>_heat_<operation>.png` | Per-operation heatmap channel (when active) |

Output directory: `outputs/rendering_<gs|mj>_<config>_<timestamp>/`

## Batched environments

When `env.batch_size > 1`, observations contain a leading environment dimension. `save_rendering.py` now exports every environment instead of only taking environment 0.

For `batch_size=1`, filenames keep the original form:

```text
<camera>_rgb.png
<camera>_depth.png
<camera>_mask.png
<camera>_overlay.png
<camera>_heatmap.png
```

For `batch_size>1`, filenames include `_env<index>` before the extension:

```text
<camera>_rgb_env0.png
<camera>_rgb_env1.png
<camera>_rgb_env2.png
<camera>_depth_env0.png
<camera>_mask_env0.png
<camera>_overlay_env0.png
<camera>_heatmap_env0.png
<camera>_heat_<operation>_env0.png
```

The same convention is used for every configured camera. Missing optional streams such as depth, mask, overlay, or heat map are reported per camera and per environment.

## Video recording

Enable the recorder to capture an MP4 or GIF of the full task rollout:

```bash
python examples/save_rendering.py +recorder.enabled=true
python examples/save_rendering.py +recorder.enabled=true +recorder.save_gif=true +recorder.fps=15
```

| Recorder option | Default | Description |
|---|---|---|
| `recorder.enabled` | false | Enable video recording |
| `recorder.fps` | 25 | Video frame rate |
| `recorder.max_steps` | 300 | Maximum steps to record |
| `recorder.save_mp4` | true | Save MP4 video |
| `recorder.save_gif` | false | Save GIF animation |
| `recorder.video_stream` | `overlay` | Frame source: `overlay`, `rgb`, or `mask` |

When recording with `env.batch_size > 1`, videos are also written per environment:

```text
<camera>_<stream>_env0.mp4
<camera>_<stream>_env1.mp4
<camera>_<stream>_env2.mp4
```

For `batch_size=1`, video filenames keep the original form:

```text
<camera>_<stream>.mp4
<camera>_<stream>.gif
```

## Interactive display

Pass `show=true` to open a matplotlib window after rendering the first frame:

```bash
python examples/save_rendering.py show=true
```
