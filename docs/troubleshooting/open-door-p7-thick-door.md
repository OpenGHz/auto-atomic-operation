# `open_door_p7_ik` 在厚门场景下推不开门排查

## 现象

`open_door_p7_ik` 默认使用 `assets/xmls/scenes/open_door/demo_p7_xf9600.xml` 时可以完成开门。

但如果把门板碰撞几何从:

```xml
<geom name="door_panel_geom" ... size="0.410000000 0.015000000 0.805000000" />
```

改成与 `assets/xmls/scenes/open_door/demo.xml` 一致的:

```xml
<geom name="door_panel_geom" ... size="0.420000000 0.020000000 0.815000000" />
```

再运行:

```bash
aao_demo --config-name open_door_p7_ik
```

会出现以下现象：

- `handle_hinge` 能短暂压下，但很快回弹
- `door_hinge` 基本保持在 `0`
- 任务卡在 `post_move` 推门阶段，最终失败

对比之下，`aao_demo --config-name open_door_airbot_play` 在同样的厚门尺寸下仍可成功。

## 结论

这不是“门变重了”或者“门锁参数不对”导致的，根因是：

- `P7 + XF9600` 这条开门轨迹对门板 **X/Z 尺寸** 很敏感
- 门板加宽、加高后，门框净空被吃掉，导致门板和门框、夹爪和门板同时进入更强碰撞
- 推门阶段还没建立有效门铰链转动，夹爪接触就先把把手挤回去，随后门锁重新挂上

换句话说，这是 **场景碰撞几何余量不足**，不是单纯的控制参数问题。

## 为什么不是质量问题

虽然 `door_panel_geom` 尺寸变大了，但门体惯量在 XML 里是显式指定的：

```xml
<inertial ... mass="6.0" diaginertia="0.2 0.2 0.2"/>
```

因此这次失败不是门体因为几何变大而“自动变重”。

## 关键排查结果

只修改门板不同尺寸分量时，结果如下：

| 变更 | 结果 | 说明 |
|---|---|---|
| 仅增厚 `Y: 0.015 -> 0.020` | 成功 | 厚度本身不是根因 |
| 仅加宽 `X: 0.410 -> 0.420` | 失败 | 左右门框余量被吃掉 |
| 仅加高 `Z: 0.805 -> 0.815` | 失败 | 上下门框余量被吃掉 |
| 同时增大 `X/Y/Z` | 失败 | 与 `demo.xml` 同尺寸时复现 |

排查结论：

- **致命因素是 `X/Z` 增大**
- `Y` 厚度增大只会让接触更紧，但单独增厚通常还能开

## 为什么 `open_door_airbot_play` 还能成功

`open_door_airbot_play` 使用的是另一套操作几何：

- 抓取基准是 `handle_grasp_*_site`
- 推门是基于抓取点快照的直推
- 整条轨迹对门板尺寸变化更鲁棒

而 `open_door_p7_ik` 的轨迹和接触关系更依赖当前 `demo_p7_xf9600.xml` 里的几何余量，因此更容易被门板宽高变化击穿。

## 直接原因

原始 `demo_p7_xf9600.xml` 里的薄门大致保留了如下净空：

- 左右各约 `15 mm`
- 上下各约 `10 mm`

当门板改成 `0.420 / 0.020 / 0.815` 后：

- 左右净空降到约 `5 mm`
- 上下净空接近 `0`

这会带来两个连锁问题：

1. 门板自身与门框更早、更强地接触
2. `XF9600` 左指垫在推门阶段更深地压进 `door_panel_geom`

结果是门还没真正绕 `door_hinge` 起转，夹爪已经把把手姿态破坏掉，`handle_hinge` 回落到门锁阈值以下，`DoorLatchCallback` 又把门锁回去。

## 推荐修复

不要只改 `door_panel_geom`，而是同时调整门框开口，保持厚门场景仍有足够碰撞余量。

本仓库最终采用的修复在 `assets/xmls/scenes/open_door/demo_p7_xf9600.xml`：

- 门板改为厚门尺寸：`0.420 0.020 0.815`
- 左右门框外移，顶部横梁抬高
- 门板、把手、`door_push_site` 整体上抬 `5 mm`

对应思路：

- 保留你需要的厚门尺寸
- 同时恢复门框净空
- 从场景几何层面解决，而不是硬调 `open_door_p7_ik` 的 waypoint

## 不推荐的方向

下面这些尝试在该问题上都不是根治：

- 仅调大把手下压角度
- 仅调小 `unlock_threshold`
- 仅回退 P7 抓取点
- 仅把门铰链 arc 改成别的 waypoint

这些办法可能短暂改善接触，但如果厚门已经把门框净空吃掉，任务仍会在推门阶段失效。

## 验证命令

修复后可直接验证：

```bash
PYTHONPATH=. /home/ghz/.mini_conda3/envs/airbot_play_data/bin/python -m auto_atom.runner.demo --config-name open_door_p7_ik env.viewer=null +max_updates=300
```

预期结果：

- `grasp_and_open` 成功
- `door_hinge` 能稳定离开 `0`
- `Summary` 中 `Success rate` 为成功

## 相关文档

- `docs/task-tuning/open_door_tuning.md`
- `docs/ik-motion-control/arc_motion_tuning.md`
