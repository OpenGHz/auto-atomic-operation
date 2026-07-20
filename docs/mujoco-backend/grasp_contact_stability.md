# 抓取时物体在夹爪里打滑/扭转（condim 与接触摩擦）

## 现象

在 `place_blocks_on_disk_airbot_play_g2`（AIRBOT Play + G2 夹爪）中，机器人抓起
积木后，从 pick 运送到 place 的过程中积木会**逐渐在夹爪里扭转、最终滑落**。任务的
成功判据比较宽松时可能仍然“成功”，但视觉上明显不稳。

## 根因分析

### `friction` 是一个三元组，但默认只用到第一项

MuJoCo geom 的 `friction="滑动 扭转 滚动"` 有三个分量：

| 分量 | 含义 | 抑制的运动 |
|------|------|-----------|
| 第 1 项（tangential） | 滑动摩擦 | 物体沿接触面**平移**滑动 |
| 第 2 项（torsional） | 扭转摩擦 | 物体绕接触法线**自转** |
| 第 3 项（rolling） | 滚动摩擦 | 物体在接触面上**滚动** |

**但后两项是否生效取决于接触的 `condim`：**

| `condim` | 生效的摩擦维度 |
|----------|---------------|
| 3（默认） | 仅滑动（第 1 项）；扭转/滚动被忽略 |
| 4 | 滑动 + 扭转 |
| 6 | 滑动 + 扭转 + 滚动 |

我们的夹爪指垫和积木都写了 `friction="1.2 0.05 0.01"`（滑动 1.2、扭转 0.05、滚动
0.01），但两者的 `condim` 都是默认的 **3**，所以 `0.05 / 0.01` 这两项**从未生效**。
两个小指垫夹住 25mm 立方体时，没有任何扭转摩擦阻止它绕夹持轴自转 —— 运送途中的姿态
变化（尤其立方体带 yaw 随机化）会让它慢慢转出来。

> 力不是瓶颈：G2 driver `forcerange=-10 10` → 约 10N 夹持力，摩擦上限 ~12N，
> 而 5g 立方体重力仅 0.05N。所以问题是**扭转自由度没被约束**，不是夹得不够紧。

### 参考对比：DISCOVERSE

`third_party/DISCOVERSE/models/mjcf/tasks_airbot_play` 的抓取很稳，关键差别就是它对
抓取接触显式设了高 `condim`：

- 夹爪手指 geom：`condim="6"`
- 被抓物体 geom：`condim="4"`
- 外加较硬的 `solref="0.01 1" solimp="2 1 0.01"`

### 实测（`place_blocks_on_disk_airbot_play_g2`，运送途中相对夹爪的姿态漂移）

| 配置 | 位置漂移 | **姿态漂移** | 任务成功 |
|------|---------|-------------|---------|
| `condim=3`（默认，修复前） | 8.6 mm | **91.5°** | 20/20 |
| `condim=6`（修复后） | 8.7 mm | **9.2°** | 20/20 |

姿态漂移从 ~90° 降到 ~9°，位置本来就是稳的（≈8.7mm 主要是夹持瞬间的落座，不是滑落）。

## 解决方法

只在**夹爪指垫**的碰撞 geom 上把 `condim` 提到 6 即可 —— MuJoCo 取接触双方
`condim` 的**较大值**，所以指垫设 6 就能让每个 “立方体↔指垫” 接触都变成 6 维，
**无需改动场景里的 10 个立方体**：

```xml
<!-- assets/xmls/robots/airbot_g2.xml -->
<default class="collision">
  <!-- condim=6 打开扭转 + 滚动摩擦；friction 的 0.05/0.01 两项在默认 condim=3
       下是失效的。 -->
  <geom contype="1" conaffinity="1" group="3" condim="6"
        friction="1.2 0.05 0.01"/>
</default>
```

因为 `airbot_g2.xml` 只被这一个任务使用，改它不会影响其他机器人/任务。

## 避坑：不要一上来就上“硬接触 + priority”

照搬 DISCOVERSE 的完整方案（在指垫上叠加 `solref="0.01 1" solimp="0.95 0.99 0.001"`
再用 `priority="2"` 让指垫接管接触参数）会把接触**过度刚化**，实测反而把成功率从
**20/20 打到 18/20**（个别抓取/释放被打飞或卡住）。

经验顺序：

1. **先只加 `condim`**（4 或 6）—— 这是治“打滑/扭转”的核心，且最稳。
2. 只有在仍不够稳时，再**小幅**调 `solref`/`solimp`，并逐步验证成功率。
3. `priority` 会让高优先级 geom **完全接管**接触的 `condim/friction/solref/solimp`，
   影响面大，非必要不用。

## 通用原则

- 抓取不稳、物体在夹爪里“转出来/滑出来”时，**先查 `condim`**，而不是先加摩擦系数。
- `friction` 第 2/3 项（扭转/滚动）只有在 `condim≥4/6` 时才有意义；写了不改 `condim`
  等于没写。
- 接触参数按“接触对”生效：`condim` 取双方**最大**，`friction` 取**逐分量最大**
  （priority 相等时）。因此把参数加在**夹爪**一侧通常最省事，一处改动覆盖所有被抓物体。
- 刚度（`solref`/`solimp`）不是越硬越好：过硬会引入弹跳/数值不稳，破坏抓取或释放。

## 排查方法

用相对夹爪的姿态漂移量化打滑（比只看位置更能暴露“扭转”型滑落）：抓稳后记录物体在
`eef_pose` 站点坐标系下的位姿作为基准，运送途中持续比较，取最大偏差；释放（物体与
`eef_pose` 距离突增）前停止统计，避免把“放下”误计成打滑。

## 受影响场景

在 `assets/xmls/robots/airbot_g2.xml` 修复（G2 夹爪，供
`place_blocks_on_disk_airbot_play_g2` 使用）。新建抓取类任务、尤其被抓物体带 yaw
随机化时，应确认夹爪接触的 `condim` 已设为 4 或 6。

相关文档：[EEF Mapper](eef_mapper.md)、[Gripper Joint Semantics](gripper_joint_semantics.md)、
[Mesh Density & Gravity](mesh_density_gravity_issue.md)。
