# Orientation Control Chain

本文档描述朝向（orientation）从 YAML 配置到 MuJoCo 仿真器的完整数据流。

## 控制方案: Mocap Body + Weld 约束

利用 MuJoCo 的约束求解器实现位姿跟踪。代码直接写入 mocap body 的位置和四元数，
weld 约束驱动物理机体跟随，无需 Euler 角转换，天然避免 gimbal lock。

## 总览

```
YAML orientation (xyzw 四元数)
  │
  ▼
resolve_orientation()          ─ 归一化四元数 / 或从 rpy Euler 转换
  │
  ▼
_resolve_pose_command()        ─ 参考系变换 → 世界坐标系 EEF 朝向
  │
  ▼
move_to_pose()                 ─ EEF → base 坐标变换
  │
  ▼
_set_mocap_pose()              ─ xyzw → wxyz 写入 data.mocap_pos/mocap_quat
  │
  ▼
env.update()                   ─ mj_step × n_substeps, weld 约束跟踪
```

## 1. YAML 配置层

```yaml
stages:
  - waypoints:
      pre_move:
        - position: [0.285, 0.004, 0.135]
          orientation: [0.500024, 0.500024, -0.499976, -0.499976]  # xyzw
          reference: object_world
```

- `orientation`: xyzw 格式四元数，表示在 `reference` 参考系下的目标朝向
- `rotation`: 可选的 rpy Euler 角（extrinsic XYZ, 即 `axes="sxyz"`），与 `orientation` 二选一
- 四元数本身没有内旋/外旋之分，它直接编码旋转矩阵

**文件**: `auto_atom/framework.py` — `PoseControlConfig`

## 2. 朝向解析

```python
# auto_atom/utils/pose.py — resolve_orientation()
def resolve_orientation(pose: PoseControlConfig) -> Orientation:
    if pose.orientation:
        return normalize_quaternion(pose.orientation)
    if pose.rotation:
        return euler_to_quaternion(pose.rotation)  # axes="sxyz"
    return (0.0, 0.0, 0.0, 1.0)  # identity
```

若使用 `rotation` 字段，通过 `quaternion_from_euler(*rpy, axes="sxyz")` 转换。`"sxyz"` 是 Gohlke transformations 库的 static(extrinsic) XYZ 约定。

## 3. 参考系变换

```python
# auto_atom/runtime.py — TaskRunner._resolve_pose_command()
reference_pose = _resolve_reference_pose(operator, pose, target)
local_pose = _pose_config_to_local_pose(pose)

if pose.relative:
    current_local = compose_pose(inverse_pose(reference_pose), current_pose)
    target_pose = compose_pose(current_local, local_pose)
else:
    target_pose = local_pose

world_pose = compose_pose(reference_pose, target_pose)
```

参考系映射:

| reference       | reference_pose                        |
| --------------- | ------------------------------------- |
| `object_world`  | 物体位置 + identity 朝向              |
| `eef_world`     | EEF 位置 + identity 朝向              |
| `eef`           | EEF 当前 full 6DOF pose               |
| `base`          | base 当前 pose                        |
| `world`         | identity pose                         |
| `object`        | 物体 full 6DOF pose                   |

输出: **世界坐标系下的 EEF 目标朝向** (xyzw 四元数)

## 4. EEF → Base 坐标变换

```python
# auto_atom/backend/mjc/mujoco_backend.py — MujocoOperatorHandler.move_to_pose()
desired_eef_pose = PoseState(position=pose.position, orientation=pose.orientation)
desired_base_pose = compose_pose(desired_eef_pose, inverse_pose(self._tool_pose_in_base))
```

`_tool_pose_in_base` 是 EEF 相对于 base link 的固定偏移（初始化时从仿真器读取）。
逆变换去掉工具偏移，得到 **世界坐标系下的 base link 目标朝向**。

## 5. 写入 Mocap Body

```python
# auto_atom/backend/mjc/mujoco_backend.py — _set_mocap_pose()
qx, qy, qz, qw = orientation_xyzw
env.data.mocap_pos[mocap_id]  = [x, y, z]
env.data.mocap_quat[mocap_id] = [qw, qx, qy, qz]  # MuJoCo uses wxyz
```

直接写入四元数，无需 Euler 角转换。MuJoCo 的 weld 约束求解器自动计算所需的
广义力，使物理机体 (`robotiq_interface`) 跟随 mocap body (`robotiq_mocap`)。

### XML 结构

```xml
<!-- 运动学目标 (mocap body) -->
<body name="robotiq_mocap" mocap="true">
  <geom type="box" size="0.01" rgba="0 1 0 0" contype="0" conaffinity="0"/>
</body>

<!-- 物理机体 (freejoint) -->
<body name="robotiq_interface" gravcomp="1">
  <freejoint name="robotiq_freejoint"/>
  ...gripper mechanism...
</body>

<!-- 约束绑定 -->
<equality>
  <weld body1="robotiq_mocap" body2="robotiq_interface" solref="0.05 1"/>
</equality>
```

### 优势

- **无 gimbal lock**: 直接使用四元数，无需 Euler 分解
- **无关节限幅**: freejoint 允许全 SO(3) 旋转
- **数值稳定**: 约束求解器处理力矩计算，避免 Jacobian 奇异点
- **代码简洁**: 不需要 `_quat_to_joint_angles()` / `_closest_angle()` 等辅助函数

## 6. 物理仿真

```python
env.update()  # mj_step × n_substeps
```

Weld 约束参数:
- `solref="0.05 1"`: 临界阻尼，50ms 时间常数
- `solimp="0.95 0.99 0.001"`: 高精度约束跟踪

### 如何调小每个仿真 step 的移动步幅

如果想让 `robotiq_interface` 每个仿真 step 跟随 `robotiq_mocap` 时走得更小、更柔和，
优先调整 `assets/xmls/robots/robotiq.xml` 中这条约束:

```xml
<weld body1="robotiq_mocap" body2="robotiq_interface" solref="0.05 1" solimp="0.95 0.99 0.001"/>
```

调参规则:

- `solref` 第 1 个值可以近似看成“收敛时间常数”
- 这个值越大，weld 跟随越慢，每个仿真 step 的位移/转角增量通常越小
- 这个值越小，weld 跟随越快，每个仿真 step 更激进，也更容易显得突然
- 第 2 个值这里保持 `1`，表示接近临界阻尼；一般先只调第 1 个值就够了

实用建议:

- 当前配置 `0.05 1`: 比 `0.02 1` 更平滑，适合作为默认值
- 如果还想更慢一点，可以试 `0.08 1` 或 `0.1 1`
- 如果觉得响应太肉、跟手性不够，可以回调到 `0.03 1` 或 `0.02 1`

建议一次只改一个量，并结合实际任务观察:

- 末端是否还有明显“跳一下”的感觉
- 接触物体时是否更稳定
- 跟随目标轨迹时是否出现明显滞后

`solimp` 通常不用先动。它主要影响约束从软到硬的响应形状；在“只是想让每 step 走小一点”
这个目标下，先调 `solref` 会更直接，也更容易预期。

## 7. 夹爪控制

```python
ctrl[0] = target_value  # 0.0 = 张开, 0.82 = 闭合
env.step(ctrl)
```

只有 1 个 actuator (`fingers_actuator`)，ctrl 向量长度为 1。

## 8. Reset / Home

Reset 时需要同步 mocap body 和 freejoint:
1. `mj_resetDataKeyframe()` 设置 freejoint qpos（物理体位置）
2. `_sync_mocap_to_freejoint()` 从 `data.xpos/xquat` 复制到 `mocap_pos/mocap_quat`
3. `home()` 恢复 mocap 和 freejoint 到快照值，零化速度

## Euler 约定速查 (参考)

| 库 / 写法 | 含义 | 等价写法 |
| --- | --- | --- |
| scipy `"ZYX"` (大写) | intrinsic ZYX (body 轴) | extrinsic xyz |
| scipy `"zyx"` (小写) | extrinsic ZYX (固定轴) | intrinsic XYZ |
| scipy `"XYZ"` (大写) | intrinsic XYZ (body 轴) | extrinsic zyx |
| scipy `"xyz"` (小写) | extrinsic XYZ (固定轴) | intrinsic ZYX |
| transformations `"szyx"` | static(extrinsic) ZYX | intrinsic XYZ |
| transformations `"rzyx"` | rotating(intrinsic) ZYX | extrinsic XYZ |
| transformations `"sxyz"` | static(extrinsic) XYZ | intrinsic ZYX |
