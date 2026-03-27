# Data Collection

This guide covers the scripts used to record task demonstrations and compare rendering outputs.

## Record Demo Video

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


## Robot Learning Data Collection

### Setup

```bash
git clone --depth 1 https://github.com/OpenGHz/AIRBOT-Data-Collection.git airdc -b develop
pip install -e ./airdc"[mujoco,assis]"
```

将`auto-atomic-operation`软链接到`third_party`目录下：

```bash
ln -s <path_to_auto_atomic_operation> airbot_ie/third_party/
```

将`操作任务配置`目录软链接到`airbot_ie/configs/managers/auto_atom`目录下：

```bash
ln -s <path_to_auto_atomic_operation>/examples/mujoco airbot_ie/configs/managers/auto_atom/mujoco
```

配置文件主要分为两部分：

- 采集环境（示教器）：`airbot_ie/configs/demonstrators/mujoco/basis`中对仿真环境进行基础配置，包括要采集的数据种类、相机的配置、仿真频率的配置等。这部分不涉及具体的任务逻辑。
- 流程管理器：`airbot_ie/configs/managers/auto_atom`中对采集流程进行配置，包括要采集哪些任务、每个任务的采集细节（如是否使用GS渲染、是否保存视频等）。这部分涉及具体的任务逻辑。

### 运行

示例命令如下：

```bash
airdc demonstrators=mujoco/basis dataset.directory=mujoco visualizer=null managers=auto_atom/pick_and_place demonstrator.component.viewer=null
```

其中：
- `demonstrators=mujoco`指定使用`MuJoCo`示教器进行数据采集。
- `dataset.directory=mujoco`指定采集的数据将被保存到`mujoco`目录下。
- `visualizer=null`不对相机的图像进行可视化。
- `managers=auto_atom/pick_and_place`指定使用`pick_and_place`任务的流程管理器进行采集。
- `demonstrator.component.viewer=null`不启动mujoco viewer。


只启动环境，不运行流程，可以使用以下命令：

```bash
airdc demonstrators=mujoco/basis dataset.directory=mujoco managers=auto_atom/pick_and_place managers.auto_atom=null
```

这将启动环境并保持空闲，等待用户通过其他命令来控制流程的运行。
