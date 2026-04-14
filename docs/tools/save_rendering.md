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

## Interactive display

Pass `show=true` to open a matplotlib window after rendering the first frame:

```bash
python examples/save_rendering.py show=true
```
