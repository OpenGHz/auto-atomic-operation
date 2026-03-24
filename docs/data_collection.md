# Data Collection

This guide covers the scripts used to record task demonstrations and compare rendering outputs.

## Record Demo

[`examples/record_demo.py`](../examples/record_demo.py) runs a task in the MuJoCo backend and saves the camera feed as a GIF and/or MP4. It reuses the same Hydra configs as `run_demo.py`.

### Basic usage

```bash
python examples/record_demo.py --config-name pick_and_place
python examples/record_demo.py --config-name cup_on_coaster
python examples/record_demo.py --config-name stack_color_blocks
python examples/record_demo.py --config-name press_three_buttons
```

Output files are written to `assets/videos/<config_name>.gif` and `assets/videos/<config_name>.mp4`.

### Recorder options

All options are injected via Hydra with the `+recorder.` prefix:

| Option | Default | Description |
| ------ | ------- | ----------- |
| `camera` | `front_cam` | Camera name to capture from |
| `fps` | `25` | Frame rate for the recording |
| `gif_width` | `320` | Width (pixels) of the output GIF |
| `save_gif` | `true` | Whether to save a GIF |
| `save_mp4` | `false` | Whether to save an MP4 |

```bash
# Save MP4 only
python examples/record_demo.py --config-name pick_and_place \
    +recorder.save_mp4=true +recorder.save_gif=false

# Use a different camera and higher FPS
python examples/record_demo.py --config-name cup_on_coaster \
    +recorder.camera=side_cam +recorder.fps=30

# Wider GIF output
python examples/record_demo.py --config-name stack_color_blocks \
    +recorder.gif_width=480
```

## Compare GS Render

[`examples/compare_gs_render.py`](../examples/compare_gs_render.py) renders the initial frame of a GS-enabled scene from every configured camera in both Gaussian Splatting and native MuJoCo modes, then saves a side-by-side comparison image.

### Basic usage

```bash
# Default config: press_three_buttons_table30_gs
python examples/compare_gs_render.py

# Specify a different GS scene config
python examples/compare_gs_render.py --config-name cup_on_coaster_table30_gs
python examples/compare_gs_render.py --config-name stack_color_blocks_table30_gs
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
