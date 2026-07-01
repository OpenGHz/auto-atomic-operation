# act_policy_eval.py 说明文档

把训练好的 lerobot **ACT** 模型接入 aao 仿真器做**闭环评估**。脚本骨架和 `policy_eval_example.py` 完全一致，区别只在于把回放录制动作的 `RecordedDemoPolicy` 换成了每步调用 ACT 模型出动作的 `ACTPolicyAdapter`。

## 数据流（单步）

```
env.capture_observation()                      # aao 仿真批量观测
  -> 拼 observation.state(10) + 两路 RGB        # 适配成 lerobot 输入
  -> policy.select_action(obs)  -> action(8)    # ACT 推理（内部带动作分块）
  -> 拆成 position(3) + quat(4) + gripper(1)     # 适配回 aao 动作
  -> env.apply_pose_action("arm", ...)          # 驱动仿真
```

## 快速开始

在 `auto-atomic-operation` 根目录下运行：

```bash
python examples/act_policy_eval.py \
    --checkpoint /path/to/pretrained_model \
    --config-name <采集训练数据时用的 aao task 配置名> \
    --batch-size 1 --num-rollouts 10
```

两个必填参数：

| 参数 | 含义 |
| --- | --- |
| `--checkpoint` | lerobot 导出的 `pretrained_model` 目录，须包含 `config.json` 与权重 |
| `--config-name` | aao task 配置名，**必须与采集训练数据时用的一致**，否则场景/相机/物体对不上 |

## 键与顺序约定（关键）

state / action / image 的键和顺序**必须与训练配置 `configs/config.yaml` 完全一致**，否则模型输入错位、评估无意义。这些约定写在脚本顶部的常量里：

- **`STATE_KEYS`** → 拼成 `observation.state`，维度 `3 + 6 + 1 = 10`
  - `arm/pose/position`（3）
  - `arm/pose/rotation_6d`（6）
  - `gripper/joint_state/position`（1）
- **`IMAGE_KEYS`** → 仿真观测的 color 键映射到模型的 image feature 名。脚本会先从 `checkpoint/config.json` 的 `input_features` 自动推断相机名（`_image_keys_from_checkpoint`），推断失败才回退到脚本内默认值。
- **动作切分**（`action(8)`，须与 `mcap.actions` 顺序一致）：
  - `ACT_POS = [0:3]` 位置
  - `ACT_QUAT = [3:7]` 姿态四元数（xyzw）
  - `ACT_GRIP = [7:8]` 夹爪（绝对值）

## 核心组件

### ACTPolicyAdapter
把 lerobot `ACTPolicy` 包成 aao `PolicyEvaluator` 需要的 policy 接口。

- `reset()`：清空 ACT 内部动作分块队列和各种 anchor，开始新一轮 rollout。
- `_build_obs()`：aao 批量观测 dict → lerobot 输入 dict。低维状态拼成 `(B,10)`；彩色图 `(B,H,W,3) uint8` 转成 `(B,3,H,W) float[0,1]`。
- `act()`：核心推理。预处理 → `select_action` → 后处理 → 拆成 `position/orientation/gripper`，按开关做相对/绝对还原和归一化，返回动作 dict。
- `check_io()`：打印一帧的训练/推理 IO 一致性报告（features、shape、dtype、范围、归一化链、missing/extra/shape mismatch）。

### 模块级函数
- `action_applier` / `observation_getter`：与 `policy_eval_example.py` 相同，负责把动作下发到 `apply_pose_action` 和抓取观测。
- `_patch_lerobot_act_attention_no_weights()`：ACT 不消费 attention 权重，模块导入时打补丁跳过那条脆弱的 CUDA 路径。
- 四元数 / rot6d 工具：`_quat_mul_batch_xyzw`、`_rot6d_to_matrix_batch`、`_global_rela_rot6d` 等，用于相对姿态还原和诊断。

## 命令行参数

### 评估规模
| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--batch-size` | 1 | 并行环境数 |
| `--num-rollouts` | 10 | 评估轮数 |
| `--max-steps` | 400 | 每轮最大步数 |
| `--sim-loop-frequency` | 10.0 | `from_config` 的仿真频率 |
| `--override` | [] | 额外 Hydra override，可重复传，例如 `--override +env=gl` |

### 动作语义开关
一次通常只开其中一种相对姿态模式。`--episode0-rela-pose` 与 chunk-relative 的 `--delta-*` 互斥（脚本会报错）。

| 参数 | 说明 |
| --- | --- |
| `--replan-every-step` | 每步清空 ACT 动作队列，基于当前观测重新预测。对 waypoint 型任务更容易从「到上方」切到「下压/闭合」 |
| `--delta-position-action` | 把 `position[0:3]` 当作相对查询时刻末端位置的 delta，再还原成绝对位置执行 |
| `--delta-orientation-action` | 把 `quat[3:7]` 当作相对 chunk 起点姿态的 delta，再还原成绝对姿态执行（chunk-relative 对比用） |
| `--episode0-rela-pose` | 按 `mcap_data_loader poses.py` 的 `_rela` 语义：pose 相对每个 rollout 第 0 帧，gripper 保持绝对值 |
| `--lock-vertical-orientation` | 忽略模型输出的 quat，强制用训练数据里的竖直末端姿态 `[0, √0.5, 0, √0.5]` |
| `--place-xy-from-target` | 诊断/仿真用：放置阶段把执行 XY 对准 `target_pedestal`，Z 和夹爪仍用模型输出 |
| `--place-xy-offset DX DY` | 配合上一项使用的目标台 XY 偏移，单位米 |

### 调试与诊断
| 参数 | 说明 |
| --- | --- |
| `--debug-action` | 打印模型动作 |
| `--action-debug-every N` | 每 N 步打印一次动作 |
| `--check-io` | 检查训练/推理输入输出 key、shape、dtype、范围和归一化链 |
| `--check-io-only` | 只做 IO 检查，不执行 rollout |
| `--trace-grasp` | 打印抓取/放置诊断：末端、方块、目标台位姿和抓取侧向误差 |
| `--trace-every N` | 开启 `--trace-grasp` 时每 N 步打印一次，阶段切换时总会打印 |

### rollout 收尾
| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--post-success-policy-seconds` | 0.0 | 所有环境判定成功后继续执行 policy 多少秒，方便在 viewer 里观察（如门是否真的继续打开） |
| `--post-done-hold-seconds` | 0.0 | rollout done 后保持 viewer/物理仿真多少秒；不再更新判定，只保留最后控制 |

## 常见用法

**先做一次 IO 一致性检查**（强烈建议接新 checkpoint 时先跑）：

```bash
python examples/act_policy_eval.py \
    --checkpoint /path/to/pretrained_model \
    --config-name my_task --check-io-only
```

**跑评估并观察抓取诊断：**

```bash
python examples/act_policy_eval.py \
    --checkpoint /path/to/pretrained_model \
    --config-name my_task \
    --num-rollouts 20 --trace-grasp --trace-every 10
```

**成功后继续运行几秒看后效：**

```bash
python examples/act_policy_eval.py \
    --checkpoint /path/to/pretrained_model \
    --config-name my_task \
    --post-success-policy-seconds 3.0
```

## 输出

每轮 rollout 结束打印：

```
[rollout 0] steps=137 completed_stages=3 success=[True]
```

全部结束打印总成功率：

```
成功率: 8/10 = 80.0%
```

## 注意事项

- `--config-name` 必须和采集训练数据时用的 aao task 配置一致。
- state / action / image 的键与顺序必须和训练配置 (`configs/config.yaml`) 完全一致，见上文 `STATE_KEYS` / `IMAGE_KEYS` / 动作切分。
- 模型输出的四元数不是单位长度，脚本在下发前会归一化（`apply_pose_action` 需要单位四元数）。
- 有 CUDA 时默认用 `cuda`，否则回退 `cpu`。
