# 相机 RGB-D 噪声模型设计

> **状态：第一阶段和第二阶段的低开销模型已实现；空间模型与真机 profile 仍为“提案，尚未实现”。**
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
- 把相机位姿、白平衡等域随机化与传感器噪声混成一个配置；
- 实现固定模式噪声、运动模糊或复杂镜头畸变；
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
          invalid_probability: 0.01
```

实现使用冻结、`extra="forbid"` 的 Pydantic 配置模型：

```python
class NoiseDistributionConfig(BaseModel, frozen=True):
    kind: Literal["gaussian", "student_t"] = "gaussian"
    degrees_of_freedom: PositiveFloat = 5.0


class TemporalNoiseConfig(BaseModel, frozen=True):
    jitter_std: NonNegativeFloat = 0.0
    ar1_coefficient: float = 0.0
    drift_std: NonNegativeFloat = 0.0
    drift_decay: float = 1.0


class RGBNoiseConfig(BaseModel, frozen=True):
    gaussian_std: NonNegativeFloat = 0.0
    shot_noise_scale: PositiveFloat | None = None
    dropout_probability: Probability = 0.0
    exposure: NonNegativeFloat = 1.0
    gain: NonNegativeFloat = 1.0
    quantization_levels: PositiveInt | None = None
    illumination_std: NonNegativeFloat = 0.0
    gain_std: NonNegativeFloat = 0.0
    distribution: NoiseDistributionConfig = NoiseDistributionConfig()
    temporal: TemporalNoiseConfig = TemporalNoiseConfig()


class DepthNoiseConfig(BaseModel, frozen=True):
    gaussian_std_m: NonNegativeFloat = 0.0
    relative_std: NonNegativeFloat = 0.0
    bias_m: float = 0.0
    bias_distance_power: NonNegativeFloat = 0.0
    scale: PositiveFloat = 1.0
    resolution_reference: tuple[int, int] | None = None
    resolution_power: float = 0.0
    quantization_step_m: PositiveFloat | None = None
    invalid_probability: Probability = 0.0
    distribution: NoiseDistributionConfig = NoiseDistributionConfig()
    temporal: TemporalNoiseConfig = TemporalNoiseConfig()


class CameraNoiseConfig(BaseModel, frozen=True):
    rgb: RGBNoiseConfig | None = None
    depth: DepthNoiseConfig | None = None
```

`noise: null` 或省略表示关闭全部噪声。字段约束至少包括：概率在 `[0, 1]`，标准
差非负，量化步长为正；未知字段应直接报错。

## 5. RGB 噪声模型（已实现）

当前支持以下可叠加效果：

| 配置 | 含义 |
| --- | --- |
| `gaussian_std` | 在归一化 RGB `[0, 1]` 输出域中加入高斯读出噪声。 |
| `shot_noise_scale` | 基于像素强度近似 Poisson 光子噪声；未设置时关闭。 |
| `dropout_probability` | 按像素随机丢失，丢失值置零。 |
| `exposure` / `gain` | 在噪声前施加静态曝光和增益。 |
| `illumination_std` / `gain_std` | 每帧采样的低频乘性变化。 |
| `quantization_levels` | 在归一化域进行量化。 |
| `distribution` | 选择 Gaussian 或标准化 Student-t 加性噪声。 |
| `temporal` | 帧级 jitter、AR(1) 和 drift；不生成空间随机场。 |

实际处理顺序为：

```text
RGB 渲染
→ 应用 rgb_clip_range_m（如已配置）
→ 转换到 [0, 1]
→ 静态 exposure/gain 与每帧 illumination/gain
→ temporal offset
→ Poisson/Shot 噪声
→ Gaussian 或 Student-t 加性噪声
→ quantization
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

## 6. Depth 噪声模型（已实现）

当前支持固定误差、距离相关误差、偏置/比例、量化、重尾分布、时序项和丢失：

| 配置 | 含义 |
| --- | --- |
| `gaussian_std_m` | 固定的米制高斯标准差。 |
| `relative_std` | 随距离增加的相对误差系数。 |
| `quantization_step_m` | 深度量化步长；省略时不量化。 |
| `invalid_probability` | 有效深度像素的随机失效概率；`dropout_probability` 仅是输入兼容别名。 |
| `bias_m` / `bias_distance_power` | 以原始距离为基准的距离相关偏置。 |
| `scale` | 深度比例误差；当配置分辨率因子时，按 `(scale - 1)` 缩放误差幅度。 |
| `resolution_reference` / `resolution_power` | 根据参考与实际像素数缩放 bias、加性误差、比例误差和 invalid probability；不生成空间随机场。 |
| `invalid_probability` | 有效深度像素被置零的概率。 |
| `distribution` / `temporal` | Student-t/Gaussian 与帧级相关噪声。 |

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
- 完整环境 `reset()` 清空时序状态并重新开始 capture 序列；批量 partial reset 只清空
  选中逻辑行的时序状态，capture 序列保持全局连续。

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

### 第二阶段：通用模型扩展（低开销部分已实现）

第二阶段先实现与具体厂商无关的运行时模型，不等待真机数据，也不要求完全复现某一
真实相机，更不把 ZED 论文参数伪装成已标定的设备 profile。可以使用一组明确标记为
`nominal` 或 `synthetic` 的近似参数，先验证配置接口、随机流、批处理、reset 以及
下游任务对噪声的响应。

这组参数的目标是产生合理的扰动范围，而不是声称复现某个真实相机。参数应放在
示意配置或测试 fixture 中，不应命名为 `zed2i_calibrated`，也不应被当作默认设备
标定结论。

当前已实现：

1. 距离/分辨率相关的 depth bias、scale 和 invalid probability；
2. Student-t 等可选重尾分布，保留 Gaussian 基线；
3. temporal jitter、AR(1) 相关噪声和慢变化 drift；
4. RGB 独立的曝光/增益、量化和低频 illumination/gain 模型。

这些模型使用人工设置的 nominal 参数即可运行，参数有效性通过单元测试、统计性质
和任务回归验证，而不是通过与某一台真机逐像素比较。RGB 与 Depth 的时序状态按
相机、流和逻辑环境行独立维护；Torch 输入直接在原设备上计算，不先转换成 NumPy。

仍未实现的高开销空间模型：

1. 空间聚集 dropout；
2. 边缘失效；
3. 低分辨率随机场或全分辨率空间滤波。

这些能力继续保留为提案，待明确性能预算和验收数据后再实现。

### 第三阶段：真机校准与 profile（提案，尚未实现）

真机采集、离线统计和 profile 拟合都可以延后到第二阶段之后。真机 profile 的作用是
把通用模型的参数替换为某个设备、固件、分辨率和环境条件下的实测值，不是运行时
模型实现或第二阶段验收的前置条件。

需要区分两个难度不同的工作：

- **离线统计评估**：读取已对齐的深度帧和真值，计算 bias、MAE、RMSE、分位数、
  invalid rate 等指标，并按距离/分辨率分组。这部分可以独立实现，主要是数据校验、
  掩码处理和常规统计计算，工程复杂度相对较低，但仍需要明确数据格式和真值契约；
- **profile 拟合与验收**：从有限实验数据选择误差分布、拟合距离曲线、估计置信区间，
  再在未参与拟合的场景上验证。它不是简单调用拟合函数，结果会受真值对齐、目标
  材质、光照、分辨率、异常帧和样本量显著影响，必须保留数据质量报告和人工复核。

只有数据覆盖足够且验证集表现稳定后，才生成版本化 profile。profile 接入时应只
替换模型参数和选择，不重写已经验证过的运行时处理架构。下面的字段和接口均为
设计方向，不属于当前可运行的任务 schema。

#### 10.1 ZED 2i 结果对后续校准的约束

`third_party/docs/camera_noise/zed.md` 总结了 ZED 2i 室内测试的以下结果：

- 测试覆盖 1--20 m、2K/1080p/720p/VGA 四种分辨率，每组连续采集 80 帧；
- 2K/1080p 的可靠深度距离约为 18 m，720p 约为 14 m，VGA 约为 7 m；
- 误差呈左偏和重尾，不能假设为简单 Gaussian；
- 明显 jitter 的起点随分辨率变化：VGA 约 4 m、720p 约 8 m、1080p/2K 约 13 m；
- RMSE 随距离近似指数增长，1080p 的置信区间相对更窄。

因此，通用模型应允许分辨率和距离参与计算，后续 profile 也应把设备的统计
可靠范围与几何裁剪范围分开。论文中的参数只能作为待拟合的先验，不能直接视为
所有 ZED、所有固件和所有室内场景的默认值。

#### 10.2 真实数据采集与校准协议

建议先定义统一的离线 RGB-D 采集记录。每条记录至少包含：

- 设备型号、固件版本、深度模式和相机序列号（可脱敏）；
- RGB/depth 分辨率、帧率、曝光、增益和其他会影响成像的设置；
- 光照条件、目标材质/纹理/颜色和场景标识；
- 目标距离及激光测距仪等外部真值，包含真值精度和单位；
- 连续 RGB/depth 帧、采集时间戳、无效像素比例和相机姿态；
- 深度无效的原始厂商表示，以及转换后的内部有效性标记。

校准集和验证集应按距离、分辨率、材质/纹理、光照和场景拆分。至少保留一组
完全没有参与拟合的场景，避免只在单一平面和单一距离上得到看似准确的参数。
外部设备适配层可以把 RealSense 的 `0`、ZED 的 `-1/NaN` 等哨兵值转换成项目
统一的零值深度契约；运行时噪声模型不应依赖厂商哨兵值。

#### 10.3 距离和分辨率相关的 depth profile

建议将有效性和误差拆成三个互不替代的 profile：

```text
depth_clip_range_m       几何/渲染的硬裁剪范围
validity_profile         设备在距离和分辨率下产生可靠深度的概率
error_profile            有效深度的偏差、尺度和误差分布
```

其中 `max_reliable_distance_m` 是统计意义上的可靠范围，不应自动覆盖或修改
`depth_clip_range_m`。候选 profile 字段包括：

- `valid_probability(distance, resolution)`；
- `max_reliable_distance_m`；
- `jitter_onset_distance_m`；
- `error_scale(distance, resolution)`；
- `outlier_probability(distance, resolution)`。

有效深度的误差建议分解为：

```text
error = systematic_bias(distance, resolution)
      + random_residual(distance, resolution, distribution)
```

尺度可以先用 `a * exp(b * distance)` 作为候选形式。分布模型按复杂度逐步支持：

1. Gaussian，作为兼容基线；
2. Student-t，处理重尾；
3. skew-t 或经验分位数模型，处理左偏和重尾同时存在的情况；
4. 混合模型，区分正常测量、离群点和失效点。

标准 Student-t 本身是对称分布，不能自动表达 ZED 文档中的左偏，因此是否需要
skew-t 或经验分布必须由独立验证集的拟合质量决定。

未来 profile 可以采用以下非执行示意（字段名仍可调整）：

```yaml
# 提案，尚未实现；不是当前任务文件的可运行配置
noise:
  profile: zed2i_indoor_1080p
  depth:
    validity:
      max_reliable_distance_m: 18.0
      jitter_onset_distance_m: 13.0
    error:
      model: student_t
      bias_m:
        type: polynomial
        coefficients: [ ... ]
      scale_m:
        type: exponential
        coefficients: [ ... ]
      degrees_of_freedom: 4.0
```

#### 10.4 空间结构与失效区域

当前 depth dropout 是逐像素 IID 基线。真实立体相机还可能在低纹理区域、物体
边缘、遮挡边界或远距离区域产生连通孔洞和大片失效。建议按以下顺序扩展：

1. 距离相关的 invalid probability；
2. block/patch dropout；
3. 边缘增强失效；
4. 连通孔洞生成；
5. 基于低分辨率随机场上采样的空间相关失效。

空间相关误差也可以先用低分辨率随机场上采样，再叠加现有 IID 噪声，避免第一步
就引入专用滤波器。所有这些模型只作用于 depth；RGB 不通过黑色像素或 depth
有效性判断来决定是否加噪。若下游需要 RGB-D 配对有效性，应在融合边界单独生成
pairing mask。

#### 10.5 时间相关 jitter 和 drift

连续帧 jitter 说明每次 capture 完全独立采样并不总是足够。候选的最小时间模型为：

```text
e_t = rho * e_(t-1) + sqrt(1-rho^2) * innovation_t
```

建议支持 AR(1) 或低阶状态空间模型，并明确以下运行时契约：

- reset 时清空时间状态；
- 每个逻辑环境行和每个相机拥有独立状态；
- shared-physics batch 不能因为共享物理场景而共享噪声状态；
- 如果采样间隔变化，相关系数应按时间间隔换算；
- 可选的慢变化 bias/drift 不应改变 mask、heat-map 等监督真值。

#### 10.6 RGB 单独校准

ZED 文档主要描述 depth，不能据此推导 RGB 参数。RGB 应使用独立的采集和拟合
流程，候选因素包括读出噪声、光子噪声、量化噪声、曝光/增益导致的亮度偏差、
白平衡/颜色增益、暗部与亮部的不同噪声水平、饱和裁剪和低频 illumination/gain
field。RGB 和 depth 继续使用独立随机流，RGB 噪声不读取 depth mask。

#### 10.7 统计验收指标

第二阶段的验证不应只检查“输出发生了变化”，而应将真实数据和仿真 profile 在
独立验证集上对齐比较：

| 通道 | 指标 |
| --- | --- |
| Depth | signed bias、MAE、RMSE、标准差、p50/p90/p95/p99 绝对误差、invalid/outlier rate、按距离和分辨率的误差曲线、连续帧 jitter、空间自相关、边缘与平面内部差异 |
| RGB | 各通道均值/方差、暗部/亮部噪声、饱和率、dropout rate、空间自相关或频谱 |

Gaussian、Student-t、skew-t 和经验分布应比较拟合质量及验证集误差，不能因为
论文推荐了 t 分布就跳过模型比较。

#### 10.8 第三阶段校准实施顺序

1. 统一真实 RGB-D 采集格式和厂商无效值转换；
2. 实现只读、可复现的离线统计评估报告，先不自动改写运行时配置；
3. 用独立验证集比较 distance/resolution-dependent bias、scale 和 validity 候选；
4. 在数据支持时加入 Student-t，并验证是否确实需要 skew-t 或经验分布；
5. 通过验收后冻结版本化设备 profile，并接入已有运行时模型；
6. 持续用新设备、固件和环境数据扩展 profile，避免把单次实验结论泛化为设备默认值。

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
