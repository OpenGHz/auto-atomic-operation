from auto_atom.basis.mjc.mujoco_basis import EnvConfig


def test_env_config_broadcasts_single_interest_operation() -> None:
    config = EnvConfig.model_validate(
        {
            "model_path": "assets/xmls/scenes/press_three_buttons/demo.xml",
            "mask_objects": ["button_blue", "button_green", "button_pink"],
            "operations": ["press"],
        }
    )

    assert config.interests == (
        ["button_blue", "button_green", "button_pink"],
        ["press", "press", "press"],
    )
