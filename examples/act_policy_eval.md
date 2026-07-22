# ACT 策略训练与评估完整流程

本文档介绍如何在 aao 仿真环境中采集数据、训练 ACT 模型、并进行闭环评估的完整流程。

## 一、数据采集

### 1.1 环境准备

确保已安装 [AIRBOT-Data-Collection](https://github.com/DISCOVER-Robotics/AIRBOT-Data-Collection)：

```bash
# 使用 Pixi（推荐）
pixi install
pixi shell -e collect

# 或使用传统方式
pip install -e .[all,airbot]
```

### 1.2 启动数据采集

在 `AIRBOT-Data-Collection` 目录下运行：

```bash
airdc --path airbot_ie/configs/aao_config.yaml dataset.directory=my_task
```

- `--path`：指定配置文件路径
- `dataset.directory`：数据保存目录名（保存在 `data/` 下）

采集过程中使用键盘控制流程，按 `i` 键查看按键说明。

### 1.3 数据说明

采集完成后，数据保存为 `.mcap` 格式文件，位于 `data/my_task/` 目录。每个 episode 对应一个 `.mcap` 文件，包含：

- **状态数据**：机械臂位姿、夹爪状态等
- **动作数据**：示教端的位姿指令
- **图像数据**：相机采集的 RGB 图像

## 二、模型训练

### 2.1 安装训练环境

确保已安装 [MCAP-DataLoader](https://github.com/OpenGHz/MCAP-DataLoader) 和 lerobot：

```bash
pip install mcap-data-loader lerobot
```

### 2.2 准备训练配置

创建训练配置文件 `configs/config.yaml`：

```yaml
batch_size: 8
num_workers: 4
policy:
  type: act
  chunk_size: 100
  n_action_steps: 100

dataset:
  root: data           # MCAP 数据根目录
  repo_id: my_task     # 与采集时 dataset.directory 一致

mcap:
  states:
    - /follow/arm/pose/position
    - /follow/arm/pose/rotation_6d
    - /follow/gripper/joint_state/position
  actions:
    - /lead/arm/pose/position
    - /lead/arm/pose/orientation
    - /lead/gripper/joint_state/position
  images:
    - /wrist_cam/color/image_raw
    - /env0_cam/color/image_raw
```

**配置说明：**

- `states`：观测状态 topic 列表，会拼接成 `observation.state`
- `actions`：动作 topic 列表，会拼接成 `action`
- `images`：图像 topic 列表，会添加到 `observation.images`

**关键要求：**

- `states` / `actions` / `images` 的顺序必须固定，训练和评估时必须完全一致
- topic 名称必须与 MCAP 数据中的 topic 名称匹配

### 2.3 开始训练

```bash
mcap-lerobot-train -c configs/config.yaml
```

训练过程中会自动：
- 加载 MCAP 数据并转换为 lerobot 格式
- 保存 checkpoint 到 `outputs/train/<timestamp>_act/checkpoints/`
- 记录训练日志

训练完成后，模型保存在 `outputs/train/<date>/<time>_act/checkpoints/last/pretrained_model/`。

## 三、闭环评估

### 3.1 数据流

评估时的数据流向：

```
env.capture_observation()                      # aao 仿真观测
  -> 拼接 state(10) + 图像                      # 转换为 lerobot 输入
  -> policy.select_action(obs)  -> action(8)   # ACT 推理
  -> 拆分为 position(3) + quat(4) + gripper(1)  # 转换为 aao 动作
  -> env.apply_pose_action("arm", ...)         # 执行动作
```

### 3.2 运行评估

在 `auto-atomic-operation` 根目录下运行：

```bash
python examples/act_policy_eval.py \
    --checkpoint /path/to/pretrained_model \
    --config-name pick_and_place \
    --num-rollouts 20
```

**必填参数：**

- `--checkpoint`：训练得到的 `pretrained_model` 目录路径
- `--config-name`：aao task 配置名，**必须与采集数据时用的配置一致**

**常用参数：**

- `--num-rollouts`：评估轮数（默认 10）
- `--batch-size`：并行环境数（默认 1）
- `--episode0-rela-pose`：使用相对姿态（推荐，与数据预处理一致）

### 3.3 IO 一致性检查

评估前建议先检查训练/推理的输入输出是否一致：

```bash
python examples/act_policy_eval.py \
    --checkpoint /path/to/pretrained_model \
    --config-name pick_and_place \
    --check-io-only
```

这会输出：
- checkpoint 的 input/output features
- 实际观测的 key、shape、dtype
- 是否有 missing/extra keys 或 shape 不匹配

### 3.4 评估输出

每轮 rollout 结束打印：

```
[rollout 0] steps=137 completed_stages=3 success=[True]
```

全部结束打印总成功率：

```
成功率: 18/20 = 90.0%
```

## 四、动作表示方式

ACT 模型支持三种动作表示方式，训练和评估时必须匹配：

### 4.1 绝对姿态（默认）

训练配置使用原始绝对 topic：

```yaml
mcap:
  actions:
    - /lead/arm/pose/position
    - /lead/arm/pose/orientation
    - /lead/gripper/joint_state/position
```

评估时**不需要**添加任何相对姿态参数。

### 4.2 Episode 相对姿态

数据预处理生成相对于每个 episode 第 0 帧的相对姿态：

```bash
python mcap_data_loader/scripts/data_process/poses.py \
  data/my_task \
  --keys /follow/arm/pose/position /follow/arm/pose/orientation \
         /lead/arm/pose/position /lead/arm/pose/orientation \
  --targets rela rotation_6d
```

训练配置使用预处理后的 topic：

```yaml
mcap:
  states:
    - /follow/arm/pose/position_rela
    - /follow/arm/pose/rotation_6d
    - /follow/gripper/joint_state/position
  actions:
    - /lead/arm/pose/position_rela
    - /lead/arm/pose/rotation_6d
    - /lead/gripper/joint_state/position
```

评估时添加 `--episode0-rela-pose` 参数：

```bash
python examples/act_policy_eval.py \
    --checkpoint /path/to/pretrained_model \
    --config-name pick_and_place \
    --episode0-rela-pose
```

### 4.3 Chunk 相对姿态

训练时使用 `ACTChunkRelativePoseProcessorStep`，将动作转换为相对于当前观测状态的增量，适合快速实验。

#### 4.3.1 准备 processor 文件

首先将 `act_relative_processor.py` 复制到训练环境可访问的位置：

```bash
# 在 auto-atomic-operation 目录
cp examples/act_relative_processor.py /path/to/training_dir/
```

#### 4.3.2 训练配置

创建训练配置 `configs/chunk_relative_config.yaml`：

```yaml
batch_size: 8
num_workers: 4

policy:
  type: act
  chunk_size: 100
  n_action_steps: 100

dataset:
  root: data
  repo_id: my_task

mcap:
  states:
    - /follow/arm/pose/position      # 3 维
    - /follow/arm/pose/rotation_6d    # 6 维（注意：需要 rotation_6d）
    - /follow/gripper/joint_state/position  # 1 维
  actions:
    - /lead/arm/pose/position         # 3 维
    - /lead/arm/pose/orientation      # 4 维（四元数 xyzw）
    - /lead/gripper/joint_state/position  # 1 维
  images:
    - /wrist_cam/color/image_raw
    - /env0_cam/color/image_raw
```

**关键要点**：
- `states` 中必须使用 `rotation_6d`（6 维），不能是 `orientation`（4 维四元数）
- `actions` 中使用 `orientation`（4 维四元数）
- state 维度：3 + 6 + 1 = 10
- action 维度：3 + 4 + 1 = 8

#### 4.3.3 注册 processor

训练前需要确保 `ACTChunkRelativePoseProcessorStep` 被注册。有两种方式：

**方式 1：在训练脚本开头导入**

在运行训练前，先导入 processor 注册它：

```bash
# 在 Python 环境中
python3 -c "import sys; sys.path.insert(0, '/path/to/training_dir'); import act_relative_processor"
```

**方式 2：修改训练启动脚本**

如果你有自己的训练启动脚本，在开头添加：

```python
import sys
sys.path.insert(0, '/path/to/training_dir')
import act_relative_processor  # 注册 processor
```

#### 4.3.4 添加 processor 到训练管道

方式有两种：

**方式 A：通过配置文件（推荐）**

如果 mcap-lerobot-train 支持在配置文件中指定 processor，添加：

```yaml
# 在 configs/chunk_relative_config.yaml 中添加
training:
  policy_preprocessor_class: act_relative_processor.ACTChunkRelativePoseProcessorStep
  policy_preprocessor_kwargs:
    relative_position: true
    relative_orientation: true
    position_slice: [0, 3]
    quat_slice: [3, 7]
    state_position_slice: [0, 3]
    state_rot6d_slice: [3, 9]
```

**方式 B：通过代码注入**

如果配置文件不支持，需要修改训练脚本，在数据加载后、训练循环前添加：

```python
from act_relative_processor import ACTChunkRelativePoseProcessorStep

# 创建 processor
chunk_relative_processor = ACTChunkRelativePoseProcessorStep(
    relative_position=True,
    relative_orientation=True,
    position_slice=(0, 3),      # action 中 position 的位置
    quat_slice=(3, 7),           # action 中 quat 的位置
    state_position_slice=(0, 3), # state 中 position 的位置
    state_rot6d_slice=(3, 9),    # state 中 rotation_6d 的位置
)

# 添加到训练管道
# 具体方式取决于你使用的训练框架
```

#### 4.3.5 开始训练

```bash
mcap-lerobot-train -c configs/chunk_relative_config.yaml
```

训练完成后，processor 会自动保存到 checkpoint 的 `policy_preprocessor.json` 文件中。

#### 4.3.6 评估

评估时添加 `--delta-position-action` 和 `--delta-orientation-action`：

```bash
python examples/act_policy_eval.py \
    --checkpoint /path/to/pretrained_model \
    --config-name pick_and_place \
    --delta-position-action \
    --delta-orientation-action \
    --num-rollouts 20
```

**自动检测**：评估脚本会自动检测 checkpoint 中是否包含 `ACTChunkRelativePoseProcessorStep`，如果参数不匹配会报错：

- 如果 checkpoint 是 chunk-relative 训练但没加 `--delta-*` 参数 → 报错
- 如果 checkpoint 不是 chunk-relative 但加了 `--delta-*` 参数 → 报错
- 如果 chunk-relative checkpoint 同时使用了 `--episode0-rela-pose` → 报错

**优点**：
- 不需要预处理数据
- 训练时自动处理相对姿态转换
- 适合快速实验和迭代

## 五、注意事项

1. **配置一致性**：评估时的 `--config-name` 必须与采集数据时用的 aao 配置一致，否则场景/相机/物体对不上
2. **topic 顺序**：训练配置中 `states` / `actions` / `images` 的顺序必须固定，评估时脚本会按相同顺序拼接
3. **动作表示匹配**：
   - 绝对姿态：不加任何参数
   - Episode 相对姿态：添加 `--episode0-rela-pose`
   - Chunk 相对姿态：添加 `--delta-position-action --delta-orientation-action`
   - 脚本会自动检测 checkpoint 训练方式，参数不匹配会报错
4. **GPU 加速**：评估时会自动使用 CUDA（如果可用），否则回退到 CPU

