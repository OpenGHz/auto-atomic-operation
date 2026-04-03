# Mesh Geom density 过高导致物体无法抬起

## 现象

运行 `aao_demo --config-name arrange_flowers_gs` 时，机器人成功抓住花并开始抬升，但抬到一定高度后停住不动，无法继续上升，最终超时失败。

## 根因分析

### Mesh Geom 的质量计算

MuJoCo 中 `type="mesh"` 的 geom 设置 `density` 后，质量 = mesh 体积 × density。当一个物体由多个 mesh part 组成时，每个 part 的质量会累加到 body 总质量。

花（flower）由 26 个 mesh geom 组成，每个都设置了 `density="100"`：

```xml
<body name="flower" pos="0.05 0.1 0.2">
  <freejoint name="flower_joint"/>
  <geom name="flower_vis_000" type="mesh" mesh="flower_part_000" density="100" contype="1" conaffinity="1" group="1"/>
  <geom name="flower_vis_001" type="mesh" mesh="flower_part_001" density="100" contype="1" conaffinity="1" group="1"/>
  <!-- ... 共 26 个 mesh part -->
</body>
```

26 个 mesh × density=100 累加后，花的总质量远超机器人 mocap weld 约束所能承受的力。

### Mocap Weld 约束的力有限

机器人通过 mocap weld 约束跟踪目标位置（`robotiq.xml`）：

```xml
<weld body1="robotiq_mocap" body2="robotiq_interface" solref="0.3 1" solimp="0.95 0.99 0.001"/>
```

`solref="0.3 1"` 定义了约束的刚度和阻尼。当抓取的物体过重时，重力产生的下拉力超过 weld 约束能提供的跟踪力，导致机器人无法将物体抬升到目标位置。

### 对比：同场景中其他物体

同一场景中的 vase 和 vase2 使用 `density="0"`（不贡献质量），不存在此问题。

### 实测质量对比

| density | 花的总质量 | 结果 |
|---------|----------|------|
| 100 | 过重（约 0.045 kg） | 抬不起来，超时 |
| 10 | 0.0045 kg | 正常抓取和放置 |
| 1 | 0.00045 kg | 抓取成功，但放置时太轻掉不下来 |

## 解决方法

将花的 mesh geom `density` 从 `100` 降低到 `10`：

```xml
<!-- 修改前 -->
<geom name="flower_vis_000" type="mesh" ... density="100" contype="1" conaffinity="1" group="1"/>

<!-- 修改后 -->
<geom name="flower_vis_000" type="mesh" ... density="10" contype="1" conaffinity="1" group="1"/>
```

需要同时修改 `demo.xml` 和 `demo_gs.xml` 中所有 26 个 flower geom。

## 通用原则

### Mesh Geom 的 density 设置

1. **多 part mesh 物体**：density 会作用于每个 geom 并累加，part 越多总质量越大。设置 density 时需考虑 part 数量。
2. **合理范围**：对于需要被 mocap weld 机器人抓取的物体，总质量应控制在几克（~0.005 kg）级别。
3. **太重**（density 过高）：mocap weld 约束力不足以克服重力，物体抬不起来。
4. **太轻**（density 过低或为 0）：物体在夹爪松开后无法靠重力脱落，导致 place 阶段检测失败（"operator is still grasping"）。

### 排查方法

用 Python 快速检查物体质量：

```python
import mujoco, os
os.chdir('assets/xmls/scenes/<scene_name>')
m = mujoco.MjModel.from_xml_path('demo_gs.xml')
for i in range(m.nbody):
    name = m.body(i).name
    if '<object>' in name:
        print(f'{name}: mass={m.body(i).mass[0]:.6f} kg')
```

## 受影响场景

此问题在 `arrange_flowers` 的 `demo.xml` 和 `demo_gs.xml` 中修复（2026-04-03）。

新建场景时若使用多 part mesh 作为可抓取物体，应注意检查 density 设置。
