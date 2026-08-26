import pytest
from pydantic import ValidationError

from auto_atom.basis.mjc.mujoco_basis import EnvConfig


_SCENE = {"base": "assets/xmls/scenes/press_three_buttons/demo.xml"}


def test_env_config_broadcasts_single_interest_operation() -> None:
    config = EnvConfig.model_validate(
        {
            "scene": _SCENE,
            "mask_objects": ["button_blue", "button_green", "button_pink"],
            "operations": ["press"],
        }
    )

    assert config.interests == (
        ["button_blue", "button_green", "button_pink"],
        ["press", "press", "press"],
    )


@pytest.mark.parametrize(
    "legacy_field", ["model_path", "robot_paths", "scene_assembly"]
)
def test_env_config_rejects_legacy_scene_fields(legacy_field: str) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EnvConfig.model_validate({"scene": _SCENE, legacy_field: "legacy"})
