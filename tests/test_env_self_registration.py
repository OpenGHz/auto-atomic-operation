from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.config_loader import compose_task_config
from auto_atom.runner.common import prepare_task_file
from auto_atom.runtime import ComponentRegistry


def test_mock_env_self_registers_from_config() -> None:
    ComponentRegistry.clear()
    cfg = compose_task_config("mock", config_dir=ROOT / "aao_configs")

    task_file = prepare_task_file(cfg)

    try:
        assert task_file.task.env_name == "mock_single_arm"
        assert ComponentRegistry.has_env("mock_single_arm")
        env = ComponentRegistry.get_env("mock_single_arm")
        assert isinstance(env, dict)
        assert env["kind"] == "mock_env"
    finally:
        ComponentRegistry.clear()
