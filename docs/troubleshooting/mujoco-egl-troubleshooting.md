# MuJoCo EGL 渲染初始化失败排查

## 现象

运行 `airdc` 时在创建 `mujoco.Renderer` 阶段崩溃:

```
RuntimeError: The MUJOCO_EGL_DEVICE_ID environment variable must be an integer
between 0 and -1 (inclusive), got 0.
```

完整栈从 `mujoco_basis.py:407` 的 `mujoco.Renderer(...)` 进入
`mujoco/egl/__init__.py:95` 的 `create_initialized_egl_device_display()`。

报错信息里 "between 0 and -1" 看起来奇怪,其实是 MuJoCo 在用 `len(all_devices)-1` 做范围提示,而 `len(all_devices) == 0`——**EGL 一个设备都没枚举到**,跟 `MUJOCO_EGL_DEVICE_ID` 是否设置无关。

## 根因

NVIDIA 的 EGL vendor 注册文件被意外删除:

```
/usr/share/glvnd/egl_vendor.d/10_nvidia.json   <-- 丢失
```

目录里只剩 `50_mesa.json`。libglvnd 按这个目录发现 EGL vendor,找不到 NVIDIA 的 JSON
就不会去加载 `libEGL_nvidia.so.0`,于是 `eglQueryDevicesEXT()` 返回 0 个设备。

`nvidia-smi` 正常、CUDA 正常、`libEGL_nvidia.so.590.48.01` 也还在 —— 但只要这个
vendor JSON 缺失,headless EGL 就用不了。

用 `dpkg -V libnvidia-gl-590:amd64` 可以列出该包所有被改动/丢失的文件,本机确认丢了 5 个:

```
missing  /usr/lib/x86_64-linux-gnu/nvidia/wine/nvngx.dll
missing  /usr/share/egl/egl_external_platform.d/15_nvidia_gbm.json
missing  /usr/share/egl/egl_external_platform.d/20_nvidia_xcb.json
missing  /usr/share/egl/egl_external_platform.d/20_nvidia_xlib.json
missing  /usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

## 修复方案

### 方案 A:重装 NVIDIA GL 包(推荐)

一次性补齐全部丢失文件:

```bash
sudo apt install --reinstall libnvidia-gl-590
```

### 方案 B:仅补 EGL vendor JSON(最小改动)

只想恢复 headless EGL:

```bash
sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json > /dev/null <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
```

注意还有 4 个文件没恢复,Wayland/Xlib/GBM 相关场景可能仍有问题,建议优先用方案 A。

### 方案 C:无 sudo 权限时的临时绕过

- 软件渲染(很慢,仅保证能跑):

  ```bash
  MUJOCO_GL=osmesa airdc --name aao_config +test=open_the_door
  ```

- 有 X display 时走 GLFW:

  ```bash
  MUJOCO_GL=glfw airdc --name aao_config +test=open_the_door
  ```

## 验证

修好后执行:

```bash
pixi run python -c "from mujoco.egl import egl_ext as EGL; print('devices:', len(EGL.eglQueryDevicesEXT()))"
```

看到非零(本机是 `devices: 5`)即说明 vendor 注册恢复。然后重跑 airdc 即可。

## 扩展问题:为什么 `devices: 5`?

`eglQueryDevicesEXT()` 返回的是**所有已注册 EGL vendor 联合枚举**出的设备,
与物理 GPU 数不是一回事。本机两张 5090 却报 5 个,详细构成:

| idx | 设备 | 来源 | 说明 |
|-----|------|------|------|
| 0   | NVIDIA GPU 0 | `libEGL_nvidia.so.0` | 5090 #0,有 `EGL_NV_device_cuda` 扩展 |
| 1   | NVIDIA GPU 1 | `libEGL_nvidia.so.0` | 5090 #1,同上 |
| 2   | `/dev/dri/card2` | `libEGL_mesa.so.0` | Mesa 从 DRM 节点枚举 |
| 3   | `/dev/dri/card1` | `libEGL_mesa.so.0` | Mesa 从 DRM 节点枚举 |
| 4   | swrast | `libEGL_mesa.so.0` | 软件光栅器(CPU 回退) |

枚举顺序通常是 NVIDIA 在前、Mesa 在后,因此 MuJoCo 默认会挑到 idx 0(NVIDIA GPU 0)。

### 查看设备详情的脚本

```python
import ctypes
libegl = ctypes.CDLL("libEGL.so.1")
libegl.eglGetProcAddress.restype = ctypes.c_void_p
libegl.eglGetProcAddress.argtypes = [ctypes.c_char_p]

def getproc(name, restype, argtypes):
    addr = libegl.eglGetProcAddress(name.encode())
    return ctypes.CFUNCTYPE(restype, *argtypes)(addr)

EGLDeviceEXT = ctypes.c_void_p
eglQueryDevicesEXT = getproc("eglQueryDevicesEXT",
    ctypes.c_int, [ctypes.c_int, ctypes.POINTER(EGLDeviceEXT), ctypes.POINTER(ctypes.c_int)])
eglQueryDeviceStringEXT = getproc("eglQueryDeviceStringEXT",
    ctypes.c_char_p, [EGLDeviceEXT, ctypes.c_int])

n = ctypes.c_int(0)
eglQueryDevicesEXT(0, None, ctypes.byref(n))
arr = (EGLDeviceEXT * n.value)()
eglQueryDevicesEXT(n.value, arr, ctypes.byref(n))

EGL_DRM_DEVICE_FILE_EXT = 0x3233
EGL_EXTENSIONS = 0x3055
for i in range(n.value):
    drm = eglQueryDeviceStringEXT(arr[i], EGL_DRM_DEVICE_FILE_EXT)
    ext = eglQueryDeviceStringEXT(arr[i], EGL_EXTENSIONS)
    print(f"[{i}] drm={drm.decode() if drm else None}")
    print(f"    ext={ext.decode() if ext else None}")
```

### 选择特定 GPU

`MUJOCO_EGL_DEVICE_ID` 是**整个 EGL 设备列表的索引**,不是 `CUDA_VISIBLE_DEVICES`
的索引。只用 GPU 1:

```bash
MUJOCO_EGL_DEVICE_ID=1 airdc --name aao_config +test=open_the_door
```

不要选到 2/3/4:2/3 是 Mesa 从 DRM 节点枚举(性能差且不一定能初始化),4 是软件渲染。

## 关键参考

- `/data/home/haizhou/airdc/.pixi/envs/default/lib/python3.12/site-packages/mujoco/egl/__init__.py:35-62` —— MuJoCo 选择 EGL 设备的逻辑
- `/usr/share/glvnd/egl_vendor.d/` —— libglvnd 扫描 EGL vendor 的目录
- `dpkg -V <pkg>` —— 检查已装包是否被改动/删除
