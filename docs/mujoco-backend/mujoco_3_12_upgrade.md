# MuJoCo 3.12 升级记录（3.6.0 → 3.12.0）

> 升级日期：2026-09-09。基线版本 3.6.0，目标版本 3.12.0（上限 `<3.13.0`）。

## 为什么升级

对照 [MuJoCo 3.7–3.12 changelog](../../third_party/docs/mujoco-changelog.md)，
本项目最值得拿的是：

- **3.12 大 mesh 凸碰撞加速**（部分 ~2×）：场景大量使用 mesh 碰撞（机器人 + UniDoor
  凸分解 + GS 扫描物体），单步加速在 batch 下按副本数放大。
- **3.8 `multiccd` 默认开启**：多点凸接触默认生效。作为行为级变化需回归验证——
  本升级通过全量测试与代表性端到端任务（dishwasher 放盘、开门/门闩、抓取）确认无
  回归后，**采纳新默认，不显式覆盖**。
- 顺带可用：3.9 PGS 确定性排序（默认 PGS 场景跨机复现性更好）。

其余特性（Flex 系、`dcmotor`/PID actuator、`surfacevel`/`adhesion`、threadpool）
对本项目当前任务形态无直接收益，未采用。

## 依赖变更

`pyproject.toml`（`[project.optional-dependencies].mujoco`）：

```toml
mujoco = ["mujoco >=3.12.0,<3.13.0", "PyOpenGL_accelerate"]
```

## 3.12 需要的代码适配

### 1. `mjt*` 枚举比较必须 `int()` 包裹（核心破坏点）

MuJoCo >= 3.8 将 `mujoco.mjt*` 成员改为 **pybind11 Enum**。虽然 `enum == int`
正常，但 **numpy 标量对枚举成员做集合成员判断会静默失效**：

```python
np.int32(3) in {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}  # False（3.12）
int(np.int32(3)) in {int(...), int(...)}                                     # True
```

症状：`DoorLatchCallback: joint 'door_hinge' must be hinge or slide`（door 系列
3 个测试文件同时失败）。

修复（与仓库既有 `int()` 风格一致）：

- `auto_atom/callbacks/door_latch.py`：`jnt_type` / `eq_type` 比较
- `auto_atom/basis/mjc/mujoco_basis.py`：`eq_type` weld 判断

> 排查建议：全仓对 `.mjt*` 枚举的比较统一走 `int()`（model_initialization.py、
> mujoco_env.py、tactile_sensor.py 等早已如此）。

### 2. 测试浮点精确断言放宽（3.12 碰撞数值噪声）

- `tests/test_xf9600_contact_parameters.py`：`contact.dist == -0.001` 原容差
  `1e-12`；3.12 凸碰撞距离带 ~1e-9 噪声（实测 5.8e-10）→ 放宽到 `1e-6`。
- `tests/test_gripper_obs_ctrl_consistency.py`：xf9600 指距（指垫**中心**欧氏距）
  在空载极限闭合处有 ~0.2 mm 极小值回弹——平行连杆过闭合时指垫越过中线所致，
  属度量伪影而非控制/观测发散（3.6 也有 +0.07 mm，仅低于旧容差 1e-4；3.12 为
  +0.19 mm）。单调容差放宽到 `5e-4` 并注释。抓取带工件时指垫停在被抓物上，
  不会出现该伪影。

## 验证结果

- 全模型 XML 在 3.12 加载 + 步进 smoke：35/35 独立模型通过（7 个“失败”均为
  include 片段/相对路径子资产，非真实失败）。
- `gaussian_renderer`（editable 安装，只依赖 mujoco 稳定 API）导入兼容。
- 全量测试 `run_tests_safe.py --test-targets tests --max-concurrency=1`：
  **61/61 通过，无失败批次**。
- `multiccd` 默认 ON 行为经端到端用例验证无回归。

## 关联提交

- `9ab9c96 chore(mujoco): pin backend to 3.12 and cover multiccd contacts`
  （pyproject 下限 + backend-contracts multiccd 覆盖）
- `64e8205 fix(mujoco): compare mjt* enums via int() for 3.12 pybind semantics`
- `2b115d8 test(mujoco): relax FP-exact tolerances broken by 3.12 collision noise`

## 未做 / 后续机会

- **复制式 batch 的多线程 step**：实测 `mj_step` 释放 GIL，B 个独立 model/data
  多线程 step 加速 B=2 ~2×、B=4 ~4×、B=8 ~5×，与顺序结果位级一致（详见仓库
  记忆 `/memories/repo/mujoco-perf.md`）。这是应用层优化、与 MuJoCo 版本无关，
  未在本升级内实现。
- Flex/软体、PID/dcmotor、surfacevel/adhesion：未来任务类型扩展时按需引入。
