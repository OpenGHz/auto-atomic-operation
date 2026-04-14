# GS 场景中碰撞盒尺寸引发的物体漂浮问题

## 现象

在 MuJoCo Viewer 中加载 `demo_gs.xml` 并点击 keyframe 后，部分带 `freejoint` 的动态物体（如 `blue_duck`、`chicken_doll`、`transparent_tape_paper`）出现以下异常：

- 物体悬浮在桌面上方（明显漂浮）
- 物体在仿真开始瞬间被弹飞（穿入桌面后被物理引擎弹出）

## 根因分析

### 坐标系差异

`demo_gs.xml` 的 keyframe 物体位置来自第三方 table30 场景（GS 采集时的真实位置），而 `demo.xml` 的 keyframe 位置是为该文件内部的桌面高度单独设置的。

两套 XML 的桌面高度不同：

| 文件 | 桌面几何中心 z | 桌面半高 | **桌面顶面 z** |
|------|-------------|---------|--------------|
| `demo.xml` | 0.03 | 0.03 | **0.06** |
| `demo_gs.xml` | 0.0642 | 0.01 | **0.0742** |

### 碰撞盒尺寸照搬问题

碰撞盒（`*_col geom`）从 `demo.xml` 照搬到 `demo_gs.xml` 时未做调整。碰撞盒 z 半高决定了物体底面位置：

```
物体底面 z = keyframe_z - col_half_height_z
```

以 `blue_duck` 为例（碰撞盒半高 0.06 → 照搬自 demo.xml）：

| 参数 | demo.xml | demo_gs.xml |
|------|----------|-------------|
| keyframe z | 0.12 | 0.0998 (GS真实位置) |
| col 半高 z | 0.06 | 0.06 (照搬，**未调整**) |
| 物体底面 z | 0.12 − 0.06 = **0.06** ✅ | 0.0998 − 0.06 = **0.0398** ❌ |
| 桌面顶面 z | 0.06 | 0.0742 |
| 结果 | 恰好落在桌面 | **穿入桌面 3.4cm → 被弹飞** |

### 各物体分析（demo_gs.xml，桌面顶面 z = 0.0742）

| 物体 | keyframe z | 旧 col 半高 | 旧底面 z | 问题 |
|------|-----------|------------|---------|------|
| blue_duck | 0.0998 | 0.06 | 0.0398 | 深穿桌面 3.4cm → 弹飞 |
| chicken_doll | 0.1052 | 0.07 | 0.0352 | 深穿桌面 3.9cm → 弹飞 |
| transparent_tape_paper | 0.1 | 0.02 | 0.080 | 悬空 5.8mm → 漂浮 |

## 解决方法

**调整碰撞盒 z 半高**，使物体底面恰好贴合桌面顶面：

```
正确 col 半高 z = keyframe_z − 桌面顶面_z
```

| 物体 | keyframe z | 新 col 半高 | 新底面 z | 结果 |
|------|-----------|------------|---------|------|
| blue_duck | 0.0998 | **0.026** | 0.0738 ≈ 0.0742 ✅ | 贴桌面 |
| chicken_doll | 0.1052 | **0.031** | 0.0742 ✅ | 贴桌面 |
| transparent_tape_paper | 0.1 | **0.026** | 0.074 ≈ 0.0742 ✅ | 贴桌面 |

**注意**：修改的是 `demo_gs.xml` 中碰撞盒的 `size` 属性，**不修改** keyframe 的 z 坐标（那是 GS 视觉对齐所必须的真实位置）。

```xml
<!-- 修改前（从 demo.xml 照搬） -->
<geom name="blue_duck_col" type="box" size="0.04 0.04 0.06" .../>
<geom name="chicken_doll_col" type="box" size="0.04 0.04 0.07" .../>
<geom name="transparent_tape_paper_col" type="box" size="0.05 0.05 0.02" .../>

<!-- 修改后（与 GS keyframe 位置对齐） -->
<geom name="blue_duck_col" type="box" size="0.04 0.04 0.026" .../>
<geom name="chicken_doll_col" type="box" size="0.04 0.04 0.031" .../>
<geom name="transparent_tape_paper_col" type="box" size="0.05 0.05 0.026" .../>
```

## 通用原则

**demo_gs.xml 中碰撞盒尺寸的正确设置方式：**

1. 确认 `demo_gs.xml` 的桌面顶面 z（`table_geom_pos_z + table_geom_half_height_z`）
2. 从 GS 原始 XML（table30 帧）或 keyframe 中取得物体中心 z
3. 计算：`col_half_height_z = object_center_z − table_top_z`
4. 将该值填入 `*_col geom` 的 `size` 第三项

**不要**直接从 `demo.xml` 照搬碰撞盒的 size —— 两套 XML 的桌面高度不同，物体 z 坐标来源也不同，碰撞盒尺寸必须独立计算。

## 受影响场景

此问题已在 `wipe_the_table/demo_gs.xml` 修复（2026-03-24）。

排查其他 `demo_gs.xml` 时，可按以下步骤快速验证：
1. 在 MuJoCo Viewer 加载 `demo_gs.xml`
2. 点击 keyframe，观察动态物体是否贴合桌面
3. 若发现漂浮或弹飞，按上述公式重新计算碰撞盒 z 半高
