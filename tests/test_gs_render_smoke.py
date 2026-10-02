"""End-to-end Gaussian-splatting capture through the ``render=gs`` config group.

Runs in the pixi ``gs`` environment with the Hugging Face GS assets present;
skipped elsewhere. It exercises the batched GS path, including the
``torch.compile`` kernels in ``gaussian_renderer`` that need a host C compiler.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")
pytest.importorskip("gaussian_renderer")

from auto_atom.config_loader import compose_task_config  # noqa: E402
from auto_atom.runner.common import prepare_task_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GS needs CUDA"),
    pytest.mark.skipif(
        not (ROOT / "assets" / "gs" / "scenes" / "cup_on_coaster").is_dir(),
        reason="GS assets not downloaded (pixi run -e gs gs-assets)",
    ),
]


def test_gs_render_captures_color_for_every_camera() -> None:
    cfg = compose_task_config(
        "cup_on_coaster",
        ["render=gs", "env.batch_size=1", "env.viewer=null"],
        ROOT / "aao_configs",
    )
    task_file = prepare_task_file(cfg)
    backend = task_file.backend(task_file.task, task_file.task_operators)
    try:
        backend.setup(task_file.task)
        backend.reset()
        observation = backend.get_env().capture_observation()
    finally:
        backend.teardown()

    for name in ("wrist_cam", "env0_cam", "env1_cam"):
        color = observation[f"{name}/color/image_raw"]["data"]
        image = np.asarray(color.cpu() if hasattr(color, "cpu") else color)
        assert image.shape == (1, 352, 640, 3), name
        # Rendered splats, not an empty frame.
        assert image.std() > 5.0, name
