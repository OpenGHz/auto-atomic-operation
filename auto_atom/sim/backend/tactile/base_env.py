from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

try:
    from .tactile_sensor import TactileSensorManager
except ImportError:
    from tactile_sensor import TactileSensorManager


class BaseEnv:
    DEFAULT_XML_PATH = "../../xml/scene_double_arm.xml"
    DEFAULT_TACTILE_XML_PATH = "../../xml/tactile_sensor.xml"

    def __init__(
        self,
        xml_path=None,
        tactile_xml_path=None,
        enable_tactile=True,
        enable_tactile_visualization=False,
    ):
        self.workspace_root = Path(__file__).resolve().parent
        self.xml_path = self._resolve_path(xml_path or self.DEFAULT_XML_PATH)
        print(self.xml_path)
        self.tactile_xml_path = self._resolve_path(
            tactile_xml_path or self.DEFAULT_TACTILE_XML_PATH
        )
        print(self.tactile_xml_path)
        self.enable_tactile = bool(enable_tactile)
        self.enable_tactile_visualization = bool(enable_tactile_visualization)

        self.model = None
        self.data = None
        self.tactile_sensor = None
        self._load_environment()

    def _resolve_path(self, path_like):
        path = Path(path_like).expanduser()
        if path.is_absolute():
            return path.resolve()

        candidate = (self.workspace_root / path).resolve()
        if candidate.exists():
            return candidate

        return (self.workspace_root / "xml" / path).resolve()

    def _load_environment(self):
        import os

        original_dir = os.getcwd()
        xml_dir = self.xml_path.parent
        try:
            os.chdir(xml_dir)
            self.model = mujoco.MjModel.from_xml_path(self.xml_path.name)
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)
        finally:
            os.chdir(original_dir)
        self._viewer = None

        if self.enable_tactile:
            self.tactile_sensor = TactileSensorManager(
                self.model,
                self.data,
                enable=self.enable_tactile_visualization,
            )

    def launch_viewer(self):
        """启动 MuJoCo passive viewer（非阻塞，与仿真同步）。"""
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self._viewer

    def reset(self, keyframe_id=0):
        if self.model.nkey > keyframe_id:
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.data

    def step(self, ctrl=None, nstep=1, render_tactile=False):
        if ctrl is not None:
            ctrl_array = np.asarray(ctrl, dtype=np.float64).reshape(-1)
            count = min(ctrl_array.shape[0], self.model.nu)
            self.data.ctrl[:count] = ctrl_array[:count]

        for _ in range(max(int(nstep), 1)):
            mujoco.mj_step(self.model, self.data)

        tactile_img = None
        if render_tactile and self.tactile_sensor is not None:
            tactile_img = self.tactile_sensor.render()

        if self._viewer is not None and self._viewer.is_running():
            if tactile_img is not None:
                H, W = tactile_img.shape[:2]
                viewport = mujoco.MjrRect(
                    self._viewer.viewport.left + self._viewer.viewport.width - W,
                    self._viewer.viewport.bottom,
                    W,
                    H,
                )
                self._viewer.set_images([(viewport, tactile_img)])
            self._viewer.sync()

        return self.data

    def get_tactile_data(self):
        if self.tactile_sensor is None:
            return None
        return self.tactile_sensor.get_data()

    def render_tactile(self):
        if self.tactile_sensor is not None:
            self.tactile_sensor.render()

    def close(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        if self.tactile_sensor is not None:
            self.tactile_sensor.close()


def main():
    """示例：初始化环境并运行简单的模拟。"""
    print("[INFO] Initializing BaseEnv...")

    env = BaseEnv(
        enable_tactile=True,
        enable_tactile_visualization=True,
    )

    print("[INFO] Environment loaded successfully")
    print(f"[INFO] Model nq: {env.model.nq}, nu: {env.model.nu}")

    # Reset environment to initial state
    env.reset()
    print("[INFO] Environment reset")

    # Launch MuJoCo viewer
    env.launch_viewer()
    print("[INFO] Viewer launched")

    # Run simulation until viewer is closed
    print("[INFO] Running simulation (close viewer window to exit)...")
    while env._viewer is not None and env._viewer.is_running():
        env.step(nstep=1, render_tactile=True)

    # Print tactile data shape once
    tactile_data = env.get_tactile_data()
    if tactile_data is not None:
        print("[INFO] Tactile data shape:")
        print(f"  - tactile: {tactile_data['tactile'].shape}")
        print(f"  - wrench: {tactile_data['wrench'].shape}")

    # Close environment
    env.close()
    print("[INFO] Environment closed")


if __name__ == "__main__":
    main()
