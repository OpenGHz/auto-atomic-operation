# 相机 RGB-D 噪声模型设计

> **状态：已实现第一阶段。**
>
> `env.cameras[].noise.rgb` 与 `.depth` 已可用于 Native MuJoCo、GS 以及批量环境。
> 本文保留后续校准和更复杂噪声模型的设计边界。

## 1. 背景与当前边界

当前 MuJoCo 相机已经支持以下能力：

- `env.cameras` 分别启用 RGB、Depth、mask 和 heat-map 输出；
- `rgb_clip_range_m` 与 `depth_clip_range_m` 为 RGB 和 Depth 提供独立的米制近远
  裁剪范围；
- Native MuJoCo 和 Gaussian Splatting（GS）都能产生 RGB/Depth 观测；
- replicated batch 和 GS shared-physics batch 具有不同的物理环境与逻辑环境映射。

这些能力描述的是几何可见性和渲染结果，不是传感器误差。仿真图像因此通常比真实
RGB-D 相机干净。噪声应作为观测层的传感器模型处理，而不是写入 XML、修改场景几何，
或混入物体/相机位姿随机化。

## 2. 设计目标

- RGB 与 Depth 可以独立配置、独立启用、独立采样随机数；
- Native MuJoCo 与 GS 使用相同的噪声语义；
- 每个相机都可以有不同参数；
- 不配置噪声时，输出保持现有行为；
- 保持现有输出契约：RGB 为 `uint8`，Depth 为浮点米制深度，mask/heat-map 为干净
  的监督真值；
- 固定任务 seed 时结果可复现，并正确支持 replicated 与 shared-physics batch；
- 无效深度不会被噪声填充，噪声越界也不会绕过 Depth clip 范围；RGB 噪声不依赖
  Depth 是否有效。

## 3. 非目标

第一版不尝试覆盖：

- 修改 MuJoCo/GS 的底层光照、材质或渲染器实现；
- 在 control tick 内改变噪声模型参数；
- 把相机位姿、曝光、白平衡等域随机化与传感器噪声混成一个配置；
- 默认实现时间相关噪声、固定模式噪声、运动模糊或复杂镜头畸变；
- 对 mask 和 heat-map 添加噪声。

## 4. 配置接口

在单个 `CameraSpec` 下使用嵌套的 `noise` 配置，避免 RGB/Depth 字段散落在
相机顶层：

```yaml
env:
  cameras:
    - name: eef_wrist_cam
      role: operator
      width: 640
      height: 352
      enable_color: true
      enable_depth: true
      rgb_clip_range_m: [0.001, 50.0]
      depth_clip_range_m: [0.10, 5.0]
      noise:
        rgb:
          gaussian_std: 0.01
          shot_noise_scale: 5000.0
          dropout_probability: 0.001
        depth:
          gaussian_std_m: 0.001
          relative_std: 0.001
          quantization_step_m: 0.001
          dropout_probability: 0.01
```

实现使用冻结、`extra="forbid"` 的 Pydantic 配置模型：

```python
class RGBNoiseConfig(BaseModel, frozen=True):
    gaussian_std: NonNegativeFloat = 0.0
    shot_noise_scale: PositiveFloat | None = None
    dropout_probability: Probability = 0.0


class DepthNoiseConfig(BaseModel, frozen=True):
    gaussian_std_m: NonNegativeFloat = 0.0
    relative_std: NonNegativeFloat = 0.0
    quantization_step_m: PositiveFloat | None = None
    dropout_probability: Probability = 0.0


class CameraNoiseConfig(BaseModel, frozen=True):
    rgb: RGBNoiseConfig | None = None
    depth: DepthNoiseConfig | None = None
```

`noise: null` 或省略表示关闭全部噪声。字段约束至少包括：概率在 `[0, 1]`，标准
差非负，量化步长为正；未知字段应直接报错。

## 5. RGB 噪声模型

第一阶段支持三个可叠加的效果：

| 配置 | 含义 |
| --- | --- |
| `gaussian_std` | 在归一化 RGB `[0, 1]` 输出域中加入高斯读出噪声。 |
| `shot_noise_scale` | 基于像素强度近似 Poisson 光子噪声；未设置时关闭。 |
| `dropout_probability` | 按像素随机丢失，丢失值置零。 |

实际处理顺序为：

```text
RGB 渲染
→ 应用 rgb_clip_range_m（如已配置）
→ 转换到 [0, 1]
→ Poisson/Shot 噪声
→ Gaussian 噪声
→ dropout
→ clip 到 [0, 1]
→ 转换为 uint8
```

RGB 没有通用的“无效像素”哨兵值，也不需要在加噪前判断像素有效性。Poisson、
Gaussian 噪声直接作用于 RGB 图像的全部像素；`dropout_probability` 独立采样丢失
mask 并将命中的像素置零。配对 Depth 是否有效不参与上述处理。

`rgb_clip_range_m` 是独立于噪声的几何裁剪功能。GS 后端当前可以借助场景距离判断
哪些 RGB 像素超出该范围，但由此产生的 `rgb_clip_mask` 只负责裁剪，不作为 RGB
噪声的输入，也不复用为 Depth 的 `depth_valid_mask`。

## 6. Depth 噪声模型

第一阶段支持固定误差、距离相关误差、量化和丢失：

| 配置 | 含义 |
| --- | --- |
| `gaussian_std_m` | 固定的米制高斯标准差。 |
| `relative_std` | 随距离增加的相对误差系数。 |
| `quantization_step_m` | 深度量化步长；省略时不量化。 |
| `dropout_probability` | 有效深度像素的随机丢失概率。 |

对有效深度 `d`，实现使用：

```text
sigma(d) = sqrt(gaussian_std_m² + (relative_std × d)²)
```

处理顺序为：

```text
Depth 渲染
→ 保留原始有效像素（有限、非零，并符合 Depth 自身的有效性规则）
→ 添加距离相关 Gaussian 噪声
→ 量化
→ 重新检查 depth_clip_range_m
→ 越界、NaN、Inf 和 dropout 统一置零
```

无效深度不能加入噪声。噪声导致深度落到 near/far 范围外时，必须再次置零，避免
噪声绕过已经配置的深度量程。

## 7. 统一处理位置

噪声应在所有渲染来源合并之后统一处理，推荐的生命周期为：

```text
Native/GS 原始渲染
→ RGB/Depth clip
→ GS 输出注入
→ batch stack 或 shared-physics broadcast
→ CameraNoiseProcessor
→ structured image message 封装
```

这要求将当前观测捕获逻辑拆成“原始观测”和“公开观测”两个阶段，例如：

```python
_capture_observation_raw()
_apply_camera_noise(observation)
capture_observation()
```

具体约束如下：

- Native 与 GS 只对最终保留的 RGB/Depth 输出加一次噪声；
- GS 生成的 RGB/Depth 必须在 GS 注入完成后再处理；
- RGB 噪声直接处理全部 RGB 像素，并独立采样 dropout mask；Depth 无效不影响
  RGB，也不向 RGB 噪声暴露 `depth_valid_mask`；
- `mask`、`heat_map`、`camera_info`、外参和时间戳不加噪声；
- 噪声在 `create_image_data` / `create_image_data_batch` 之前执行；
- `to_numpy=False` 时保留既有 Torch 输出契约，避免隐式设备拷贝。

shared-physics 模式尤其重要：物理环境只有一份，但逻辑 batch 的每一行仍应使用
独立噪声。因此不能在物理行上先加噪声再广播；必须在逻辑 batch 展开之后按行处理。

## 8. 随机性与可复现性

噪声随机数不能直接复用物体位姿随机化的顺序 RNG，否则一次观测捕获会改变下一次
reset 的随机结果。

实现采用：

- `task.seed` 作为默认根种子；
- 从 `reset_index`、`capture_index`、逻辑环境索引、相机稳定 ID 和 stream ID
  （RGB/Depth）派生独立子流；
- 不使用 Python 内置 `hash()`，相机名称 ID 应使用稳定编码；
- 非零固定 seed 时逐帧可复现；默认无固定 seed 时保持非确定性；
- 每次 `capture_observation()` 默认视为一次新的传感器曝光，重复捕获生成新噪声。

replicated batch 中每个物理环境独立采样；shared-physics batch 中每个逻辑环境行
独立采样，即使其几何深度相同。

## 9. 观测与元数据契约

- RGB 继续输出 `uint8`，范围 `[0, 255]`；
- Depth 继续输出米制浮点数组；
- 无效 Depth 统一使用现有的零值约定；
- mask 和 heat-map 保持无噪声，作为监督真值；
- `get_info()["cameras"][name]` 可增加序列化后的 `noise` 配置；
- 不将内部随机计数器写入观测或相机标定信息。

## 10. 实施阶段

### 第一阶段：基础模型

1. 增加 `RGBNoiseConfig`、`DepthNoiseConfig` 和 `CameraNoiseConfig`。
2. 实现 NumPy/Torch 兼容的 `CameraNoiseProcessor`。
3. 重构 Native、GS、replicated batch 和 shared-physics batch 的观测边界，保证
   噪声只执行一次。
4. 加入 `get_info()` 配置回显和任务 schema 文档。

### 第二阶段：验证与校准

1. 对照真实 RGB-D 设备标定 `gaussian_std`、Poisson scale、深度误差和量化步长。
2. 增加小规模统计测试和固定 seed 的逐像素回归测试。
3. 根据实际需求再考虑固定模式噪声、时间相关噪声、深度离群点和 RGB 色彩增益
   偏差。

## 11. 验证要求

实现后至少应覆盖：

- 无噪声配置与当前输出保持一致；
- 固定 seed 下重复运行结果一致；
- 不同相机、环境和 RGB/Depth stream 的随机流相互独立；
- shared-physics 的不同逻辑行具有不同噪声；
- 无效 Depth 不被填充，越界 Depth 被置零；
- Depth 无效但 RGB 有效的像素仍保留 RGB 并可正常注入 RGB 噪声；
- 黑色 RGB 像素与其他颜色一样参与 RGB 噪声，不能被当成无效值跳过；
- Native 与 GS 结果遵守相同处理顺序；
- Torch、NumPy、structured 和非 structured 输出契约一致；
- mask/heat-map 不受噪声影响；
- 非法概率、负标准差和非正量化步长在配置校验阶段失败。

相关的当前配置和相机裁剪语义见
[`task_file_schema.md`](../task-configuration/task_file_schema.md)；相机位姿随机化
仍属于 [`randomization.md`](../task-configuration/randomization.md)，不应与本文的
传感器噪声混用。
