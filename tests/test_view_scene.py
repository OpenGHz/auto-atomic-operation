from __future__ import annotations

from types import SimpleNamespace

import mujoco
from omegaconf import OmegaConf

from auto_atom.backend.mjc.mujoco_backend import MujocoTaskBackend
from auto_atom.backend.mjc import viewer as viewer_module
from examples import view_scene


class _FakeEnv:
    batch_size = 1

    def __init__(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody><body name="root"><geom type="sphere" size="0.01"/></body></worldbody>
            </mujoco>
            """
        )
        self.envs = [SimpleNamespace(model=model, data=mujoco.MjData(model))]
        self.config = SimpleNamespace(
            viewer=None,
            viewer_env_index=0,
            initial_joint_positions=None,
            gaussian_render=None,
        )
        self.reset_calls = 0
        self.update_calls = 0
        self.refresh_calls = 0
        self.closed = False

    def reset(self, _env_mask=None) -> None:
        self.reset_calls += 1

    def update(self) -> None:
        self.update_calls += 1

    def refresh_viewer(self) -> None:
        self.refresh_calls += 1

    def close(self) -> None:
        self.closed = True


def _backend() -> tuple[MujocoTaskBackend, _FakeEnv]:
    env = _FakeEnv()
    return MujocoTaskBackend(env=env, operator_handlers={}, object_handlers={}), env


def test_load_backend_disables_embedded_viewer_and_runs_canonical_reset(
    monkeypatch,
) -> None:
    backend, env = _backend()
    task_file = SimpleNamespace(task=SimpleNamespace())
    seen = {}

    def prepare(config):
        seen["viewer"] = config.env.viewer
        return task_file

    monkeypatch.setattr(view_scene, "prepare_task_file", prepare)
    monkeypatch.setattr(
        view_scene,
        "construct_scene_backend",
        lambda *_args, **_kwargs: backend,
    )

    cfg = OmegaConf.create({"env": {"viewer": {"distance": 1.0}}})
    loaded = view_scene._load_backend(cfg)

    assert loaded is backend
    assert seen["viewer"] is None
    assert env.reset_calls == 1
    assert env.refresh_calls == 1


def test_without_embedded_viewer_does_not_mutate_hydra_config() -> None:
    cfg = OmegaConf.create({"env": {"viewer": {"distance": 1.0}}})

    isolated = view_scene._without_embedded_viewer(cfg)

    assert cfg.env.viewer.distance == 1.0
    assert isolated.env.viewer is None


def test_native_viewer_reload_replaces_and_tears_down_backend(monkeypatch) -> None:
    current, current_env = _backend()
    replacement, replacement_env = _backend()
    loaded = {}
    callbacks = []

    def launch(*, loader):
        callbacks.append(loader)
        model, data = loader()
        loaded["initial_model"] = model
        loaded["initial_data"] = data
        model, data = loader()
        loaded["reloaded_model"] = model
        loaded["reloaded_data"] = data

    monkeypatch.setattr(viewer_module.mujoco.viewer, "launch", launch)

    active = view_scene.run_native_viewer(current, lambda: replacement)

    assert active is replacement
    assert loaded["initial_model"] is current_env.envs[0].model
    assert loaded["initial_data"] is current_env.envs[0].data
    assert loaded["reloaded_model"] is replacement_env.envs[0].model
    assert loaded["reloaded_data"] is replacement_env.envs[0].data
    assert current_env.closed
    assert not replacement_env.closed


def test_gaussian_config_comes_from_backend_environment() -> None:
    backend, env = _backend()
    assert view_scene.gaussian_config(backend) is None

    env.config.gaussian_render = SimpleNamespace(
        body_gaussians={"root": "root.ply"},
        background_ply=None,
    )

    assert view_scene.gaussian_config(backend) is env.config.gaussian_render
