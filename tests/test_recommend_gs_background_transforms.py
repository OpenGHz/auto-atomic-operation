import numpy as np

from gaussian_renderer.core.gaussiandata import GaussianData
from gaussian_renderer.core.util_gau import save_ply

from examples.recommend_gs_background_transforms import estimate_background_offset


def test_estimate_background_offset_recovers_tabletop_center(tmp_path):
    rng = np.random.default_rng(0)

    tabletop = np.column_stack(
        [
            rng.normal(loc=1.2, scale=0.08, size=4000),
            rng.normal(loc=-0.3, scale=0.08, size=4000),
            rng.normal(loc=0.01, scale=0.004, size=4000),
        ]
    ).astype(np.float32)
    clutter = np.column_stack(
        [
            rng.normal(loc=-1.0, scale=0.5, size=1000),
            rng.normal(loc=2.0, scale=0.5, size=1000),
            rng.normal(loc=0.8, scale=0.2, size=1000),
        ]
    ).astype(np.float32)
    xyz = np.concatenate([tabletop, clutter], axis=0)

    path = tmp_path / "synthetic_bg.ply"
    save_ply(
        GaussianData(
            xyz=xyz,
            rot=np.tile(
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(xyz), 1)
            ),
            scale=np.ones((len(xyz), 3), dtype=np.float32),
            opacity=np.full((len(xyz),), 0.5, dtype=np.float32),
            sh=np.zeros((len(xyz), 3), dtype=np.float32),
        ),
        path,
    )

    estimate = estimate_background_offset(
        path,
        target_center_xy=(0.45, 0.06),
        target_tabletop_z=0.0742,
        histogram_bins=200,
        top_dense_bins=20,
        min_local_points=1000,
    )

    np.testing.assert_allclose(
        estimate.recommended_offset,
        (-0.75, 0.36, 0.0642),
        atol=0.03,
    )
