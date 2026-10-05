# microwave_sweet_potato 随机化一览

本文汇总 `task=microwave_sweet_potato`（UMI v3 夹爪把地瓜放进微波炉）每次 reset 时随机化的内容、范围和取值依据。配置文件以仓库中的 YAML 为准，各范围的推导过程和完整验证数据见 [Microwave and sweet-potato assets](../migration-notes/microwave_sweet_potato_assets.md#randomization-presets)。

## 三种配置

| 配置 | 命令 | 重力 | 特点 |
|---|---|---|---|
| 默认 | `aao-demo task=microwave_sweet_potato` | 有 | 只对夹爪起始位姿和地瓜位置做小幅抖动 |
| `gravity` | `aao-demo task=microwave_sweet_potato randomization=microwave_sweet_potato/gravity` | 有 | 地瓜躺在台面上，大范围随机 |
| `zero_gravity` | `aao-demo task=microwave_sweet_potato randomization=microwave_sweet_potato/zero_gravity` | 无（`env.gravity: [0, 0, 0]`） | 地瓜悬浮，姿态任意，按相机视野约束 |

两个预设都位于 `aao_configs/randomization/microwave_sweet_potato/`，共享 `_microwave.yaml`（微波炉位置和门）和 `_cameras.yaml`（相机）。

## 总览

| # | 随机项 | 默认 | `gravity` | `zero_gravity` |
|---|---|---|---|---|
| 1 | 微波炉门开角 | – | 90°–152° | 90°–152° |
| 2 | 微波炉在台面上的位置 | – | ✓ | ✓ |
| 3 | 地瓜初始位置 | ±3 cm | 台面上，微波炉前方 | 悬浮，相机视野内任意处 |
| 4 | 地瓜初始姿态 | – | 只绕竖直轴转 | 任意自旋 + 航向 + 仰角 |
| 5 | 夹爪起始位姿 | 小幅抖动 | 跟随地瓜 | 只跟随地瓜位置 |
| 6 | 抓取时夹爪相对地瓜的姿态 | – | 0–20° 下倾 | 绕长轴任意滚转 + 0–20° 下倾 |
| 7 | 沿长轴的抓取位置 | – | – | 偏后 0–2 cm |
| 8 | 前置相机 `env1_cam` 朝向 | – | ✓ | ✓ |
| 9 | 腕部相机 `eef_wrist_cam` 安装偏移 | – | ✓ | ✓ |

所有范围都是均匀分布。

## 各项详情

坐标约定：世界 +Y 指向微波炉（腔体朝 -Y 开口），+Z 向上。地瓜本体 x 轴是长轴，静止时指向微波炉；钝端朝向操作者。

### 1. 微波炉门开角

| 配置键 | 范围 | 含义 |
|---|---|---|
| `task.randomization.joints.microwave_door_hinge` | `[-2.0943, -1.0123]` rad | 门开 90°（与正面垂直）到 152°（铰链限位） |

- 取值为铰链关节的绝对值：0.5585 rad 为关门，向负值方向打开。
- 下限的依据：开角小于约 85° 时门的自由边会挡在腔口前，夹爪或地瓜会撞门。门固定在 44°/55°/66° 时，成功率只有 1–2/20、10–12/20、16/20。
- 门在每次 reset 最先写入，之后的位姿采样和碰撞、视野检查都基于这个开角。

### 2. 微波炉位置

| 配置键 | 范围 | 含义 |
|---|---|---|
| `entities.microwave` | x -10/+4 cm，y -3/0 cm，航向 ±15°（`yaw: ±0.26` rad） | 相对默认位置在台面上平移、转动 |

- 微波炉的支脚在所有取值下都留在台面上。
- 微波炉不参与碰撞检测（`collision_radius: 0`），因为地瓜以它为参考摆放。
- 放置目标 `microwave_target` 是微波炉的子物体，随微波炉一起移动。

### 3–4. 地瓜初始位姿

地瓜以微波炉为参考（`reference: microwave`）：先随微波炉的平移和转动一起移动，再叠加下表的偏移（世界坐标轴方向）。因此地瓜总在腔口前方。

| 配置 | 位置偏移 | 姿态偏移 | 约束 |
|---|---|---|---|
| 默认 | x、y 各 ±3 cm（相对自身默认位置，不跟随微波炉） | – | 碰撞半径 8.5 cm |
| `gravity` | x -20/+4 cm，y +1/+5 cm | 航向 ±30° | 碰撞半径 8.5 cm |
| `zero_gravity` | x -25/+45 cm，y -20/+5 cm，高于静止高度 10–40 cm | 绕长轴自旋 ±180°，长轴仰角 ±20°，航向 ±30° | 碰撞半径 8.5 cm；在 `env1_cam` 视野内（包围球，留边 8 px） |

- `gravity`：只绕竖直轴转，保持地瓜在台面上的稳定静止姿态（约 31° 滚转），所以 reset 后不会滚动。
- `zero_gravity`：不要求在台面上方。候选框比台面大，由视野约束裁剪；候选框本身只保证不碰台面（下限高度）、柜体（+y 上限）和打开的门（-x 下限）。
  - 300 次 reset 实测：91 次地瓜在台面投影范围以外。
  - 地瓜网格离画面边缘至少 25 px（约束只保证包围球留边 8 px）。
  - 离台面至少 8.8 cm，离柜体和门的碰撞体分别至少 8.3 cm、11.2 cm。

### 5. 夹爪起始位姿

| 配置 | 参考 | 位置偏移 | 姿态偏移 |
|---|---|---|---|
| 默认 | 自身默认起始位姿 | x ±5 cm，y ±4 cm，z -2/+4 cm | – |
| `gravity` | 地瓜（完整位姿跟随） | x ±6 cm，y -4/+5 cm，z -2/+8 cm | 滚转、俯仰 ±7°，航向 ±11° |
| `zero_gravity` | 地瓜（只跟随位置，`follow: position`） | 同上 | 滚转 ±45°，俯仰 ±7°，航向 ±11° |

- 碰撞半径 15 cm，与地瓜互相排斥。
- 实际起始位置（夹爪 `eef_pose` 相对地瓜，300 次 reset 实测）：
  - `gravity`：沿地瓜朝向在其后方 19–31 cm，侧向 ±7 cm，上方 10–20 cm。
  - `zero_gravity`：世界 -Y 方向 20–29 cm，侧向 ±6 cm，上方 10–20 cm。
- `zero_gravity` 只跟随位置，是因为地瓜自旋任意：完整跟随时夹爪会绕长轴公转，40 次 reset 中 15 次起始在地瓜下方、19 次倒置。
- `zero_gravity` 的 ±45° 滚转只影响起始姿态；抓取时的夹爪滚转由第 6 项决定。

### 6. 抓取时夹爪相对地瓜的姿态

两个预设都在静态参考帧 `sweet_potato_grasp_frame` 中写抓取路点，并在其中固定夹爪的完整姿态（`pick_orientation: level`）。随机化通过移动这个参考帧实现：它跟随地瓜，再按下表转动；转动都绕下爪接触点进行。

| 配置 | 随机项 | 配置键 | 范围 |
|---|---|---|---|
| `gravity` | 绕夹爪闭合轴下倾 | `entities.sweet_potato_grasp_frame.roll` | `[-0.349, 0]`：0–20° 低头 |
| `zero_gravity` | 绕长轴的夹爪滚转 | `entities.sweet_potato_grasp_roll_frame.roll`（`reference: absolute_world`） | 世界坐标下 ±90°：从水平到竖直任意 |
| `zero_gravity` | 绕夹爪闭合轴下倾 | `entities.sweet_potato_grasp_frame.roll`（参考 `sweet_potato_grasp_roll_frame`） | `[-0.349, 0]`：0–20° 低头 |

- `gravity` 的夹爪始终水平闭合：张开的下爪离台面仅约 3 mm，按夹爪几何估算，滚转 10° 下爪会降低约 9 mm，插进台面。
- `zero_gravity` 的滚转是世界坐标下的绝对值；地瓜自旋均匀，所以相对地瓜的滚转覆盖整圈。
  - 放置时夹爪可以一左一右或一上一下，所以任何滚转离较近的那种姿势都不超过 15°。范围原为 ±45°，那时放置只能一左一右。
  - 若改为相对地瓜固定滚转，夹爪会以任意世界滚转到达腔口，放置前需要转正最多 180°。150 次中 13 次掉落。
- 倾斜只取低头方向：放入前地瓜会被放平，夹爪的俯仰与抓取倾角相同。
  - 抬头约 7° 起，连杆的渲染网格就会插进腔体顶部，最多 27 mm。
  - 低头到约 30° 才会碰到。
- 上限 20°：超过后夹持在斜截面上容易打滑。
  - `gravity` 下，20–30° 的 46 次中掉 4 次。
  - `zero_gravity` 下，20–35° 的 61 次中掉 4 次。

### 7. 沿长轴的抓取位置（仅 `zero_gravity`）

| 配置键 | 范围 | 含义 |
|---|---|---|
| `microwave_sweet_potato.grasp_randomization`（抓取路点 `pre_move[2].randomization`） | `y: [-0.02, 0]` | 闭合位置向钝端偏后 0–2 cm |

上限 2 cm：地瓜后段在中心后 5–7 cm 处从 42 mm 收窄到 26 mm，夹在细段上会把悬浮的地瓜挤出去。放到 3 cm 时 29 次中丢 3 次。`gravity` 下没有试过这一项。

### 8. 前置相机 `env1_cam`

| 配置键 | 范围 |
|---|---|
| `cameras.env1_cam` | 滚转 ±4.6°（`±0.08` rad），俯仰 ±5.7°（`±0.10`），航向 ±8.6°（`±0.15`） |

- 只转动，相机位置固定。偏移叠加在相机的世界 roll/pitch/yaw 上：roll 使视野上下倾斜，yaw 主要左右平移，pitch 主要使画面在图像平面内旋转。
- 在这些极值与微波炉位置的所有组合下，柜体正面和 `gravity` 地瓜的静止区域都至少在画面内 12 px。
- 放置目标点在随机化后的画面内至少 103 px（两个预设各 200 次 reset）。

### 9. 腕部相机 `eef_wrist_cam`

| 配置键 | 范围 |
|---|---|
| `cameras.eef_wrist_cam` | 安装偏移：x ±3 mm，y -1/+5 mm，z -5/+2 mm（相机自身坐标：x 右、y 上、z 后）；绕三轴各 ±5°（`±0.087` rad） |

- 随机的是相机相对夹爪的安装偏移，不是世界位姿。
- 位置范围受夹爪外壳和手指限制：在所有 64 个极值组合下，两根手指在画面中至少保留默认面积的 53%，镜头不进入外壳。

## 采样顺序与约束

每次 reset 的写入顺序（`gravity` / `zero_gravity`）：

1. 门铰链
2. 前置相机、腕部相机安装偏移
3. 微波炉
4. 地瓜
5. 抓取参考帧（`zero_gravity` 先滚转帧，再倾斜帧）
6. 夹爪起始位姿

夹爪引用了地瓜，所以排在所有物体之后。默认配置只有夹爪和地瓜两项，按此顺序写入。

约束：

- **碰撞排斥**：地瓜与夹爪的包围球（半径 8.5 cm、15 cm）不得重叠，重叠则重采样。微波炉和抓取参考帧不参与。
- **视野约束**（仅 `zero_gravity`）：地瓜必须在 `env1_cam` 画面内，只能用固定相机做这项检查。夹爪起始位姿在地瓜之后采样，腕部相机此时还没定位，检查腕部相机会被配置校验拒绝。
- 门、微波炉和相机不做可行性检查；它们的范围按任务可行性预先选定（见各项）。
- 候选在 100 次尝试内都不满足约束时，reset 报错，不会静默使用不合规的样本。

## 不随机的部分

- 微波炉内的放置位置：目标位于转盘中心前约 3 cm，地瓜前端在柜体正面以内约 2 mm。
- 放置姿态：
  - `gravity` 和默认配置固定完整的静止姿态。
  - `zero_gravity` 不随机，但也不固定：只要求长轴水平，夹爪两指一左一右或一上一下。在此前提下取离到达时姿态最近、且沿放入、松开、退出都不碰微波炉的那个，所以随抓取和地瓜初始姿态而变（见[放置姿态自动求解](../design/nearest-feasible-place-orientation.md)）。
  - 一上一下时夹爪竖直伸展更大，所以放入和释放都抬高 30 mm，并且只张开到夹爪指令 0.008（两指间隙 60 mm）。地瓜因此停在名义目标上方 42–48 mm，一左一右时为 5–20 mm。
  - 即便如此，一上一下只在约三分之一的抓取下放得下；实际约 12% 的 episode 选中它。
- 地瓜的形状、质量（0.2 kg）和摩擦，以及台面、材质和光照。
- 相机内参和前置相机的位置。
- 传感器噪声：本任务未配置相机噪声。

## 随机种子

`task.seed` 默认为 42。同一种子每次运行得到相同的样本序列，同一次运行中各次 reset 的样本不同。需要不同序列时传 `task.seed=<n>`。

## 验证

以下为批大小 1 的端到端仿真统计，每个种子 20 个 episode：

| 配置 | 成功 | 说明 |
|---|---|---|
| 默认 | 60/60（3 个种子） | 夹爪离微波炉至少 15.6 mm |
| `gravity` | 57/60（3 个种子） | 3 次失败都是送入腔体途中地瓜从夹爪中掉落；夹爪离微波炉至少 15.9 mm |
| `zero_gravity` | 120/120（6 个种子） | 一上一下 11 次；放置阶段旋转角中位数 22°、最大 66°；夹爪离微波炉至少 3.5 mm |

没有夹爪或地瓜碰门，也没有夹爪碰柜体或台面。

`tests/test_microwave_sweet_potato_umi_v3.py` 对两个预设各跑两个有种子的 episode。它检查：

- 相机集合、门开角范围和重力设置
- 夹爪起始在地瓜上方
- 悬浮地瓜在前置相机画面内
- `zero_gravity` 的抓取参考帧倾斜不超过 20°，放置姿势按所选姿势抬高和松开，地瓜停在所选释放位置 25 mm 内
- 两次 reset 之间微波炉、门和相机确实发生了变化

## 相关文件

- `aao_configs/task/microwave_sweet_potato.yaml`：任务、默认随机化和可调参数（`microwave_sweet_potato.*`）
- `aao_configs/randomization/microwave_sweet_potato/`：`gravity.yaml`、`zero_gravity.yaml`、`_microwave.yaml`、`_cameras.yaml`
- `assets/xmls/scenes/microwave_sweet_potato/demo.xml`：放置目标、抓取点 `sweet_potato_grasp_site`、静态参考帧 `sweet_potato_grasp_frame` / `sweet_potato_grasp_roll_frame`
- [Randomization](../task-configuration/randomization.md)：随机化配置语义（参考、`follow`、`joints`、`visible_in` 等）
