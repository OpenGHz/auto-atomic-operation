"""UniDoor contract tests through the generic scene-composition seam."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from auto_atom.runner.common import prepare_task_file
from auto_atom.runtime import TaskRunner
from auto_atom.scene_composition import (
    AssetAssemblyLayerConfig,
    MjcfLayerConfig,
    SceneConfig,
    TransformConfig,
    compile_scene,
    load_composed_scene,
)
from auto_atom.utils.pose import PoseState, compose_pose


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _collision_part(
    name: str,
    vertices: list[list[float]],
    faces: list[list[int]],
    *,
    enabled: bool,
) -> dict:
    geometry = {"vertices_m": vertices, "faces": faces}
    return {
        "name": name,
        "collision_enabled": enabled,
        **geometry,
        "geometry_sha256": _json_sha256(geometry),
    }


def _write_obj(path: Path, *, scale: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                f"v {scale} 0 0",
                f"v 0 {scale} 0",
                f"v 0 0 {scale}",
                "f 1 2 3",
                "f 1 2 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _manifest_ref(root: Path, rel: str, asset_id: str, kind: str) -> dict:
    path = root / rel
    return {
        "asset_id": asset_id,
        "kind": kind,
        "manifest": rel,
        "manifest_sha256": _sha256(path),
        "status": "pass",
    }


@pytest.fixture()
def tiny_catalog(tmp_path: Path) -> Path:
    """Create the smallest valid right-hinge catalog used by unit tests."""

    root = tmp_path / "catalog"
    door_dir = root / "components" / "doors" / "D001"
    handle_dir = root / "components" / "handles" / "H003"
    lock_handle_dir = root / "components" / "handles" / "HL001"
    for path in (
        door_dir / "frame.obj",
        door_dir / "panel.obj",
        handle_dir / "handle.obj",
        lock_handle_dir / "handle.obj",
        lock_handle_dir / "lock.obj",
    ):
        _write_obj(path)

    door_manifest = {
        "asset_id": "D001",
        "kind": "door",
        "status": "pass",
        "handedness": {"hinge_side": "right"},
        "geometry": {
            "root_translation_m": [0.0, 0.0, 0.0],
            "handle_position_m": [0.8, 0.0, 0.5],
            "panel_bounds_m": [[0.0, -0.03, 0.0], [1.0, 0.03, 1.0]],
            "frame_bounds_m": [[-0.1, -0.05, -0.1], [1.1, 0.05, 1.1]],
            "frame_collision_bands": {
                "left": {
                    "position_m": [-0.05, 0.0, 0.5],
                    "half_size_m": [0.05, 0.05, 0.6],
                },
                "right": {
                    "position_m": [1.05, 0.0, 0.5],
                    "half_size_m": [0.05, 0.05, 0.6],
                },
                "top": {
                    "position_m": [0.5, 0.0, 1.05],
                    "half_size_m": [0.55, 0.05, 0.05],
                },
            },
        },
        "outputs": {
            "frame": {"path": "components/doors/D001/frame.obj"},
            "panel": {"path": "components/doors/D001/panel.obj"},
        },
    }
    handle_manifest = {
        "asset_id": "H003",
        "kind": "handle",
        "status": "pass",
        "handedness": {"hinge_side": "right"},
        "has_lock_mesh": False,
        "geometry": {
            "grasp_offset_m": [0.1, 0.0, 0.0],
            "handle_bounds_m": [[0.0, -0.03, -0.03], [0.2, 0.03, 0.03]],
        },
        "outputs": {"handle": {"path": "components/handles/H003/handle.obj"}},
    }
    lock_handle_manifest = {
        "asset_id": "HL001",
        "kind": "handle",
        "status": "pass",
        "handedness": {"hinge_side": "right"},
        "has_lock_mesh": True,
        "geometry": {
            "grasp_offset_m": [0.1, 0.0, 0.0],
            "handle_bounds_m": [[0.0, -0.03, -0.03], [0.2, 0.03, 0.03]],
        },
        "outputs": {
            "handle": {"path": "components/handles/HL001/handle.obj"},
            "lock": {"path": "components/handles/HL001/lock.obj"},
        },
    }
    door_manifest["outputs"]["frame"]["sha256"] = _sha256(door_dir / "frame.obj")
    door_manifest["outputs"]["panel"]["sha256"] = _sha256(door_dir / "panel.obj")
    handle_manifest["outputs"]["handle"]["sha256"] = _sha256(handle_dir / "handle.obj")
    lock_handle_manifest["outputs"]["handle"]["sha256"] = _sha256(
        lock_handle_dir / "handle.obj"
    )
    lock_handle_manifest["outputs"]["lock"]["sha256"] = _sha256(
        lock_handle_dir / "lock.obj"
    )
    (door_dir / "component.json").write_text(
        json.dumps(door_manifest), encoding="utf-8"
    )
    (handle_dir / "component.json").write_text(
        json.dumps(handle_manifest), encoding="utf-8"
    )
    (lock_handle_dir / "component.json").write_text(
        json.dumps(lock_handle_manifest), encoding="utf-8"
    )

    product = {
        "schema_version": "1.0",
        "combination_space": {"handedness": "right"},
        "components": {
            "doors": [
                _manifest_ref(
                    root,
                    "components/doors/D001/component.json",
                    "D001",
                    "door",
                )
            ],
            "handles": [
                _manifest_ref(
                    root,
                    "components/handles/H003/component.json",
                    "H003",
                    "handle",
                ),
                _manifest_ref(
                    root,
                    "components/handles/HL001/component.json",
                    "HL001",
                    "handle",
                ),
            ],
        },
    }
    (root / "product_space.json").write_text(json.dumps(product), encoding="utf-8")
    descriptor = {
        "schema": "aao.scene-asset-package/v1",
        "package_id": "tiny_unidoor",
        "revision": "test-1",
        "payload_root": {"uri": "."},
        "units": {"length": "m", "angle": "rad", "mass": "kg"},
        "canonical_frame": {
            "name": "package_local",
            "up": "+z",
            "handedness": "right",
            "quaternion": "xyzw",
            "transform_baked": True,
        },
        "components": {
            "source": "product_space.json",
            "index": "product_space.json",
            "selection_roles": ["door", "handle"],
        },
        "assembly_templates": [
            {
                "id": "lever_door_v1",
                "adapter": "unidoor.lever_door@1",
                "joint_axes": {
                    "door": [0.0, 0.0, -1.0],
                    "handle": [0.0, 1.0, 0.0],
                },
                "joint_specs": {
                    "door": {
                        "range": [0.0, 1.5079644737231006],
                        "springref": 0.0,
                        "stiffness": 1.0,
                        "damping": 1.2,
                        "frictionloss": 0.2,
                        "armature": 0.02,
                        "limited": True,
                    },
                    "handle": {
                        "range": [0.0, 0.65],
                        "springref": 0.0,
                        "stiffness": 3.0,
                        "damping": 0.08,
                        "armature": 0.002,
                        "limited": True,
                    },
                },
            }
        ],
        "integrity": {"algorithm": "sha256", "policy": "selected-artifacts"},
        "provenance": {},
        "extensions": {},
    }
    (root / "scene_asset_package.json").write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    return root


@pytest.fixture()
def tiny_acd_catalog(tiny_catalog: Path) -> Path:
    """Add a two-part ACD sidecar without changing the AABB fallback fixture."""

    handle_dir = tiny_catalog / "components" / "handles" / "H003"
    component_path = handle_dir / "component.json"
    component = json.loads(component_path.read_text(encoding="utf-8"))
    source_mesh_sha256 = component["outputs"]["handle"]["sha256"]
    component["source_id"] = "tiny-h003"

    tetrahedron_vertices = [
        [0.0, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.0, 0.02, 0.0],
        [0.0, 0.0, 0.02],
    ]
    tetrahedron_faces = [
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ]
    bipyramid_vertices = [
        [0.04, 0.0, 0.0],
        [0.06, 0.0, 0.0],
        [0.05, 0.02, 0.0],
        [0.05, 0.01, 0.02],
        [0.05, 0.01, -0.02],
    ]
    bipyramid_faces = [
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
        [1, 0, 4],
        [2, 1, 4],
        [0, 2, 4],
    ]
    disabled_vertices = [
        [0.0, 0.0, 0.0],
        [0.001, 0.0, 0.0],
        [0.0, 0.001, 0.0],
        [0.0, 0.0, 0.001],
    ]
    parts = [
        _collision_part(
            "part_000",
            tetrahedron_vertices,
            tetrahedron_faces,
            enabled=True,
        ),
        _collision_part(
            "part_001",
            bipyramid_vertices,
            bipyramid_faces,
            enabled=True,
        ),
    ]
    parts.extend(
        _collision_part(
            f"part_{index:03d}",
            disabled_vertices,
            tetrahedron_faces,
            enabled=False,
        )
        for index in range(2, 42)
    )
    enabled_geometry_hashes = [
        part["geometry_sha256"] for part in parts if part["collision_enabled"]
    ]
    artifact_manifest_sha256 = "a" * 64
    artifact_collection_sha256 = "b" * 64
    artifact_asset_set_sha256 = _json_sha256(
        {
            "asset_id": "H003",
            "source_mesh_sha256": source_mesh_sha256,
            "part_geometry_sha256": enabled_geometry_hashes,
        }
    )
    sidecar = {
        "schema_version": "1.0",
        "package_kind": "unidoor_component_collision_supplement",
        "representation": "motrixsim_acd_convex_parts_v1",
        "model_version": "unidoor-handle-runtime-motrixsim-acd-v1",
        "asset_id": "H003",
        "asset_kind": "handle",
        "source_id": "tiny-h003",
        "source_mesh_sha256": source_mesh_sha256,
        "handedness": {
            "hinge_side": "right",
            "matrix": [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "transform_id": "door-hinge-x-reflection-v1",
            "identity_sha256": (
                "d629f199482168c681d87aae599aebfad4e1c41316018a6a8aaca9a9c7a90a8d"
            ),
        },
        "actual_part_count": 2,
        "topology_slot_count": 42,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "artifact_collection_sha256": artifact_collection_sha256,
        "artifact_asset_set_sha256": artifact_asset_set_sha256,
        "parts": parts,
    }
    sidecar_path = handle_dir / "collision" / "motrixsim_acd_v1.json"
    sidecar_path.parent.mkdir(parents=True)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    component["collision_supplement"] = {
        "path": "components/handles/H003/collision/motrixsim_acd_v1.json",
        "sha256": _sha256(sidecar_path),
        "schema_version": "1.0",
        "representation": "motrixsim_acd_convex_parts_v1",
        "asset_id": "H003",
        "asset_kind": "handle",
        "source_mesh_sha256": source_mesh_sha256,
        "actual_part_count": 2,
        "topology_slot_count": 42,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "artifact_collection_sha256": artifact_collection_sha256,
        "artifact_asset_set_sha256": artifact_asset_set_sha256,
        "hinge_side": "right",
    }
    component_path.write_text(json.dumps(component), encoding="utf-8")

    product_path = tiny_catalog / "product_space.json"
    product = json.loads(product_path.read_text(encoding="utf-8"))
    handle_reference = next(
        reference
        for reference in product["components"]["handles"]
        if reference["asset_id"] == "H003"
    )
    handle_reference["manifest_sha256"] = _sha256(component_path)
    product_path.write_text(json.dumps(product), encoding="utf-8")
    return tiny_catalog


def _write_host(path: Path) -> None:
    path.write_text(
        '<mujoco model="host"><worldbody><body name="host"/></worldbody></mujoco>',
        encoding="utf-8",
    )


def _config(
    root: Path,
    *,
    package: Path | None = None,
    host: Path | None = None,
    robot: Path | None = None,
    door: str = "D001",
    handle: str = "H003",
    namespace: str = "door",
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    verify_hashes: bool = True,
) -> SceneConfig:
    host = host or root / "host.xml"
    if not host.exists():
        _write_host(host)
    layers: list[AssetAssemblyLayerConfig | MjcfLayerConfig] = [
        AssetAssemblyLayerConfig(
            package=package or root,
            adapter="unidoor.lever_door@1",
            selection={"door": door, "handle": handle},
            namespace=namespace,
            placement=TransformConfig(
                position=position,
                orientation_xyzw=orientation,
            ),
            verify_hashes=verify_hashes,
        )
    ]
    if robot is not None:
        layers.append(MjcfLayerConfig(path=robot))
    return SceneConfig(base=host, layers=tuple(layers))


def test_build_tiny_unidoor_scene_contains_contract_names(tiny_catalog: Path) -> None:
    mujoco = pytest.importorskip("mujoco")

    artifact = compile_scene(_config(tiny_catalog))
    model = mujoco.MjModel.from_xml_string(artifact.xml)

    assert model.nbody >= 4
    assert model.ngeom >= 7
    assert (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "door__unidoor_assembly")
        >= 0
    )
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "door__door_hinge") >= 0
    assert (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "door__handle_hinge") >= 0
    )
    assert (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "door__handle_grasp_center")
        >= 0
    )
    latch_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, "door__door_latch_lock"
    )
    assert latch_id >= 0
    assert model.eq_type[latch_id] == mujoco.mjtEq.mjEQ_JOINT
    assert bool(model.eq_active0[latch_id])
    assert (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "door__door_latch") < 0
    )
    assert artifact.semantic_refs["door.door.hinge.joint"] == "door__door_hinge"
    assert artifact.semantic_refs["door.latch.constraint"] == "door__door_latch_lock"


def test_handle_collision_falls_back_to_aabb_without_supplement(
    tiny_catalog: Path,
) -> None:
    mujoco = pytest.importorskip("mujoco")

    model = mujoco.MjModel.from_xml_string(compile_scene(_config(tiny_catalog)).xml)
    handle_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "door__door_handle"
    )
    collision_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == handle_body_id
        and int(model.geom_contype[geom_id]) != 0
    ]

    assert len(collision_geom_ids) == 1
    collision_geom_id = collision_geom_ids[0]
    assert model.geom_type[collision_geom_id] == mujoco.mjtGeom.mjGEOM_BOX
    assert model.geom_size[collision_geom_id].tolist() == pytest.approx(
        [0.1, 0.03, 0.03]
    )
    assert model.body_mass[handle_body_id] == pytest.approx(0.25)
    expected_inertia = [0.00015, 0.0009083333333333334, 0.0009083333333333334]
    assert model.body_inertia[handle_body_id].tolist() == pytest.approx(
        expected_inertia
    )


def test_handle_acd_parts_are_loaded_as_convex_mesh_geoms(
    tiny_acd_catalog: Path,
) -> None:
    mujoco = pytest.importorskip("mujoco")

    artifact = compile_scene(_config(tiny_acd_catalog))
    model = mujoco.MjModel.from_xml_string(artifact.xml)
    handle_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "door__door_handle"
    )
    collision_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == handle_body_id
        and int(model.geom_contype[geom_id]) != 0
    ]

    # Only the two enabled slots become collision geoms; the 40 fixed-topology
    # placeholders remain provenance data and never enter the MuJoCo model.
    assert len(collision_geom_ids) == 2
    assert all(
        model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        for geom_id in collision_geom_ids
    )
    assert all(int(model.geom_contype[geom_id]) == 16 for geom_id in collision_geom_ids)
    assert all(
        int(model.geom_conaffinity[geom_id]) == 6 for geom_id in collision_geom_ids
    )
    mesh_ids = [int(model.geom_dataid[geom_id]) for geom_id in collision_geom_ids]
    assert len(set(mesh_ids)) == 2
    assert sorted(
        (int(model.mesh_vertnum[mesh_id]), int(model.mesh_facenum[mesh_id]))
        for mesh_id in mesh_ids
    ) == [(4, 4), (5, 6)]
    assert model.body_mass[handle_body_id] == pytest.approx(0.25)
    expected_inertia = [0.00015, 0.0009083333333333334, 0.0009083333333333334]
    assert model.body_inertia[handle_body_id].tolist() == pytest.approx(
        expected_inertia
    )

    visual_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "door__door_handle_visual"
    )
    assert model.geom_type[visual_geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    assert int(model.geom_contype[visual_geom_id]) == 0
    assert int(model.geom_conaffinity[visual_geom_id]) == 0
    sidecar_path = (
        tiny_acd_catalog
        / "components"
        / "handles"
        / "H003"
        / "collision"
        / "motrixsim_acd_v1.json"
    )
    assert sidecar_path.resolve() in artifact.dependencies


def test_handle_collision_supplement_hash_mismatch_is_rejected(
    tiny_acd_catalog: Path,
) -> None:
    sidecar_path = (
        tiny_acd_catalog
        / "components"
        / "handles"
        / "H003"
        / "collision"
        / "motrixsim_acd_v1.json"
    )
    sidecar_path.write_text(
        sidecar_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="collision supplement sha256 mismatch"):
        compile_scene(_config(tiny_acd_catalog))


def test_handle_collision_supplement_invalid_face_is_rejected(
    tiny_acd_catalog: Path,
) -> None:
    sidecar_path = (
        tiny_acd_catalog
        / "components"
        / "handles"
        / "H003"
        / "collision"
        / "motrixsim_acd_v1.json"
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["parts"][0]["faces"][0][0] = len(sidecar["parts"][0]["vertices_m"])
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid face index"):
        compile_scene(_config(tiny_acd_catalog, verify_hashes=False))


def test_handle_collision_supplement_unknown_representation_is_rejected(
    tiny_acd_catalog: Path,
) -> None:
    sidecar_path = (
        tiny_acd_catalog
        / "components"
        / "handles"
        / "H003"
        / "collision"
        / "motrixsim_acd_v1.json"
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["representation"] = "unknown_collision_v1"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="representation mismatch"):
        compile_scene(_config(tiny_acd_catalog, verify_hashes=False))


def test_real_h004_uses_all_declared_acd_parts_if_assets_are_available() -> None:
    mujoco = pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[1]
    catalog = root / "third_party" / "unidoor_lever_catalog_pipeline_right_hinge"
    package = (
        root
        / "assets"
        / "scene_assets"
        / "unidoor_lever_right_hinge"
        / "scene_asset_package.json"
    )
    host = root / "assets" / "xmls" / "scenes" / "open_door_unidoor" / "demo.xml"
    sidecar_path = (
        catalog
        / "components"
        / "handles"
        / "H004"
        / "collision"
        / "motrixsim_acd_v1.json"
    )
    if not package.is_file() or not host.is_file() or not sidecar_path.is_file():
        pytest.skip("local UniDoor H004 collision assets are unavailable")

    artifact = compile_scene(
        _config(
            catalog,
            package=package,
            host=host,
            handle="H004",
        )
    )
    model = mujoco.MjModel.from_xml_string(artifact.xml)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    enabled_parts = [part for part in sidecar["parts"] if part["collision_enabled"]]
    handle_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "door__door_handle"
    )
    collision_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == handle_body_id
        and int(model.geom_contype[geom_id]) != 0
    ]

    assert sidecar["actual_part_count"] == len(enabled_parts) == 14
    assert len(collision_geom_ids) == len(enabled_parts)
    assert all(
        model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        for geom_id in collision_geom_ids
    )
    assert sidecar_path.resolve() in artifact.dependencies


def test_xyzw_orientation_is_emitted_as_mjcf_wxyz(tiny_catalog: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    config = _config(
        tiny_catalog,
        position=(1.0, 2.0, 3.0),
        orientation=(0.0, 0.0, 0.707106781, 0.707106781),
    )
    model = mujoco.MjModel.from_xml_string(compile_scene(config).xml)
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "door__unidoor_assembly"
    )
    assert model.body_pos[body_id].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert model.body_quat[body_id].tolist() == pytest.approx(
        [0.707106781, 0.0, 0.0, 0.707106781]
    )


def test_loader_keeps_host_relative_mesh_paths(
    tmp_path: Path, tiny_catalog: Path
) -> None:
    host = tmp_path / "host.xml"
    _write_obj(tmp_path / "host.obj", scale=0.25)
    host.write_text(
        """
<mujoco model="host">
  <asset><mesh name="host_mesh" file="host.obj"/></asset>
  <worldbody><geom name="host_geom" type="mesh" mesh="host_mesh"/></worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    mujoco = pytest.importorskip("mujoco")
    model = load_composed_scene(_config(tiny_catalog, host=host))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "host_mesh") >= 0


def test_host_with_existing_door_contract_is_rejected(
    tmp_path: Path, tiny_catalog: Path
) -> None:
    host = tmp_path / "host.xml"
    host.write_text(
        """
<mujoco model="host">
  <worldbody><body name="old_door"><joint name="door__door_hinge"/></body></worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="joint/door__door_hinge"):
        compile_scene(_config(tiny_catalog, host=host))


def test_missing_component_id_is_rejected(tiny_catalog: Path) -> None:
    with pytest.raises(KeyError, match="D999"):
        compile_scene(_config(tiny_catalog, door="D999"))


def test_raw_catalog_directory_is_rejected(tiny_catalog: Path) -> None:
    (tiny_catalog / "scene_asset_package.json").unlink()

    with pytest.raises(FileNotFoundError, match="scene_asset_package.json"):
        compile_scene(_config(tiny_catalog))


def test_hash_mismatch_is_rejected(tiny_catalog: Path) -> None:
    manifest = tiny_catalog / "components" / "doors" / "D001" / "component.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        compile_scene(_config(tiny_catalog))


def test_manifest_path_cannot_escape_catalog_root(tiny_catalog: Path) -> None:
    product_path = tiny_catalog / "product_space.json"
    product = json.loads(product_path.read_text(encoding="utf-8"))
    product["components"]["doors"][0]["manifest"] = "../outside.json"
    product_path.write_text(json.dumps(product), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes catalog root"):
        compile_scene(_config(tiny_catalog, verify_hashes=False))


def test_handedness_mismatch_is_rejected(tiny_catalog: Path) -> None:
    manifest = tiny_catalog / "components" / "handles" / "H003" / "component.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["handedness"]["hinge_side"] = "left"
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="hinge-side mismatch"):
        compile_scene(_config(tiny_catalog, verify_hashes=False))


def test_lock_handle_is_visual_only(tiny_catalog: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        compile_scene(_config(tiny_catalog, handle="HL001")).xml
    )
    lock_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "door__door_lock")
    lock_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "door__door_lock_visual"
    )
    assert lock_body >= 0
    assert lock_geom >= 0
    assert int(model.geom_contype[lock_geom]) == 0
    assert int(model.geom_conaffinity[lock_geom]) == 0


def test_real_catalog_host_and_robot_compile_if_assets_are_available() -> None:
    mujoco = pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[1]
    catalog = root / "third_party" / "unidoor_lever_catalog_pipeline_right_hinge"
    package = (
        root
        / "assets"
        / "scene_assets"
        / "unidoor_lever_right_hinge"
        / "scene_asset_package.json"
    )
    host = root / "assets" / "xmls" / "scenes" / "open_door_unidoor" / "demo.xml"
    robot = root / "assets" / "xmls" / "robots" / "p7_arm_v3_with_umi_gripper_v3.xml"
    if (
        not (catalog / "product_space.json").is_file()
        or not package.is_file()
        or not host.is_file()
        or not robot.is_file()
    ):
        pytest.skip("local UniDoor catalog/host/robot assets are unavailable")

    config = _config(
        catalog,
        package=package,
        host=host,
        robot=robot,
        position=(1.54, 0.79, -1.0),
        orientation=(0.0, 0.0, 0.707106781, 0.707106781),
    )
    model = load_composed_scene(config)
    assert model.ncam == 4
    assert (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "door__unidoor_assembly")
        >= 0
    )
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "door__door_handle") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "handle_body_phys") < 0

    # The pre-generated combination is a structural regression oracle for the
    # selected component pair; the AAO host adds the robot and cameras around
    # the same canonical hinge/site names.
    combo = catalog / "combinations" / "D001-H003" / "D001-H003.xml"
    if combo.is_file():
        oracle = mujoco.MjModel.from_xml_path(str(combo))
        for joint_name in ("door_hinge", "handle_hinge"):
            actual_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"door__{joint_name}"
            )
            oracle_id = mujoco.mj_name2id(oracle, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            assert model.jnt_axis[actual_id].tolist() == pytest.approx(
                oracle.jnt_axis[oracle_id].tolist()
            )


def test_demo_places_handle_and_robot_on_the_same_door_side() -> None:
    mujoco = pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "aao_configs"
    if not (
        root
        / "third_party"
        / "unidoor_lever_catalog_pipeline_right_hinge"
        / "product_space.json"
    ).is_file():
        pytest.skip("local UniDoor catalog is unavailable")

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="open_door_unidoor_p7_v3_umi_v3")
    scene = SceneConfig.model_validate(
        OmegaConf.to_container(config.env.scene, resolve=True)
    )
    model = load_composed_scene(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assembly_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "door__unidoor_assembly"
    )
    robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "p7_mount")
    grasp_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "door__handle_grasp_center"
    )
    handle_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "door__handle_hinge"
    )
    panel_normal = data.xmat[assembly_id].reshape(3, 3)[:, 1]
    robot_side = float(
        np.dot(data.xpos[robot_id] - data.xpos[assembly_id], panel_normal)
    )
    grasp_side = float(
        np.dot(data.site_xpos[grasp_id] - data.xpos[assembly_id], panel_normal)
    )

    assert robot_side < 0.0
    assert grasp_side < 0.0
    assert data.xaxis[handle_joint_id].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert list(config.task.stages[1].param.post_move[0].arc.axis) == [1, 0, 0]


def test_demo_final_approach_targets_the_explicit_grasp_site() -> None:
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "aao_configs")):
        config = compose(config_name="open_door_unidoor_p7_v3_umi_v3")

    pick_stage, pull_stage, push_stage = config.task.stages
    assert pick_stage.name == "pick_handle"
    assert pick_stage.operation == "pick"
    assert pick_stage.site == "door__handle_grasp_center"
    assert list(pick_stage.param.pre_move[-1].position) == [0.02, 0.0, 0.0]
    assert pick_stage.param.pre_move[-1].tolerance.position == pytest.approx(0.004)
    assert (
        pick_stage.param.pre_move[-1].tolerance.position
        < config.task_operators.arm.control.tolerance.position
    )
    assert config.env.initial_joint_positions is None
    assert dict(config.task_operators.arm.initial_state.joint_positions) == {
        "joint1": 0.0,
        "joint2": -0.8,
        "joint3": 0.0,
        "joint4": -1.2,
        "joint5": 0.0,
        "joint6": 0.0,
        "joint7": 0.0,
    }
    assert config.task_operators.arm.initial_state.eef == pytest.approx(0.0)
    base_pose = config.task_operators.arm.initial_state.base_pose
    # The pose defaults to world; only the position component is anchored to
    # the selected handle, with its z axis explicitly overridden to world.
    assert base_pose.get("reference", "world") == "world"
    assert base_pose.position.reference == "door__handle_grasp_center"
    assert base_pose.position.x == pytest.approx(0.2474)
    assert base_pose.position.y == pytest.approx(-0.4666)
    assert base_pose.position.z.value == pytest.approx(-0.019236672160874)
    assert base_pose.position.z.reference == "world"
    assert base_pose.orientation.get("reference", "world") == "world"
    assert base_pose.orientation.roll == pytest.approx(0.0)
    assert base_pose.orientation.pitch == pytest.approx(1.5707963267948966)
    assert base_pose.orientation.yaw == pytest.approx(0.0)
    base_randomization = config.task.randomization.arm.base
    assert base_randomization.reference == "relative"
    assert list(base_randomization.x) == pytest.approx([-0.01, 0.01])
    assert list(base_randomization.y) == pytest.approx([-0.01, 0.01])
    assert list(base_randomization.z) == pytest.approx([-0.005, 0.005])
    assert list(base_randomization.roll) == pytest.approx([-0.02, 0.02])
    assert list(base_randomization.pitch) == pytest.approx([-0.02, 0.02])
    assert list(base_randomization.yaw) == pytest.approx([-0.03, 0.03])
    assert "eef" not in pick_stage.param
    assert pull_stage.name == "pull_handle"
    assert pull_stage.operation == "pull"
    assert "site" not in pull_stage
    assert "pre_move" not in pull_stage.param
    assert "eef" not in pull_stage.param
    assert len(pull_stage.param.post_move) == 1
    assert pull_stage.param.post_move[0].arc.pivot == "door__handle_hinge"
    assert push_stage.name == "push_open"
    assert push_stage.operation == "push"
    assert "site" not in push_stage
    assert len(push_stage.param.pre_move) == 1
    assert push_stage.param.pre_move[0].arc.pivot == "door__door_hinge"
    assert "post_move" not in push_stage.param


def test_operator_base_pose_can_be_anchored_to_a_named_scene_frame() -> None:
    """A base override composes a fixed local transform with the grasp frame."""
    mujoco = pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[1]
    if not (
        root
        / "third_party"
        / "unidoor_lever_catalog_pipeline_right_hinge"
        / "product_space.json"
    ).is_file():
        pytest.skip("local UniDoor catalog is unavailable")

    with initialize_config_dir(
        version_base=None,
        config_dir=str(root / "aao_configs"),
    ):
        config = compose(
            config_name="open_door_unidoor_p7_v3_umi_v3",
            overrides=[
                "door_id=D001",
                "handle_id=HL016",
                "env.batch_size=1",
                "env.cameras=[]",
                "env.enabled_sensors=[]",
                "env.viewer=null",
                "task.randomization.arm.base=null",
            ],
        )
    with open_dict(config):
        config.task_operators.arm.initial_state = {
            "base_pose": {
                "reference": "door__handle_grasp_center",
                "position": [0.0, 0.5, -0.2],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            }
        }

    runner = TaskRunner().from_config(prepare_task_file(config))
    try:
        backend = runner._context.backend
        frame = backend.get_element_pose("door__handle_grasp_center", 0)
        expected = compose_pose(
            frame,
            PoseState(position=[0.0, 0.5, -0.2], orientation=[0.0, 0.0, 0.0, 1.0]),
        )
        actual = backend.get_operator_handler("arm").get_base_pose().select(0)
        np.testing.assert_allclose(actual.position, expected.position, atol=1e-6)
        np.testing.assert_allclose(actual.orientation, expected.orientation, atol=1e-6)

        # The frame is sampled at reset/setup, not followed while the door is
        # articulated during execution.
        before = actual.position.copy()
        update = runner.reset()
        del update
        env = backend.get_env().envs[0]
        hinge_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "door__door_hinge",
        )
        env.data.qpos[env.model.jnt_qposadr[hinge_id]] = 0.2
        mujoco.mj_forward(env.model, env.data)
        after = backend.get_operator_handler("arm").get_base_pose().select(0)
        np.testing.assert_allclose(after.position, before, atol=1e-6)
    finally:
        runner.close()


def test_demo_grasps_before_unlatching_and_unlocks_before_opening() -> None:
    mujoco = pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[1]
    if not (
        root
        / "third_party"
        / "unidoor_lever_catalog_pipeline_right_hinge"
        / "product_space.json"
    ).is_file():
        pytest.skip("local UniDoor catalog is unavailable")

    with initialize_config_dir(
        version_base=None,
        config_dir=str(root / "aao_configs"),
    ):
        config = compose(
            config_name="open_door_unidoor_p7_v3_umi_v3",
            overrides=[
                "env.batch_size=1",
                "env.cameras=[]",
                "env.enabled_sensors=[]",
                "env.viewer=null",
            ],
        )
    runner = TaskRunner().from_config(prepare_task_file(config))
    try:
        backend = runner._context.backend
        single_env = backend.get_env().envs[0]
        latch_id = mujoco.mj_name2id(
            single_env.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            "door__door_latch_lock",
        )
        saw_pull_effect = False
        saw_push_effect = False
        update = runner.reset()
        for (
            joint_name,
            expected,
        ) in config.task_operators.arm.initial_state.joint_positions.items():
            joint_id = mujoco.mj_name2id(
                single_env.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            assert joint_id >= 0
            qpos_address = int(single_env.model.jnt_qposadr[joint_id])
            assert float(single_env.data.qpos[qpos_address]) == pytest.approx(
                float(expected)
            )
        for _ in range(500):
            update = runner.update()
            active = runner._env_states[0].active
            if active is None:
                if bool(update.done[0]):
                    break
                continue
            if (
                active.plan.stage.name == "pull_handle"
                and active.action_index == 1
                and not saw_pull_effect
            ):
                saw_pull_effect = True
                assert backend.is_object_grasped(
                    "arm", "door__door_handle"
                ).tolist() == [True]
                assert bool(single_env.data.eq_active[latch_id])
            if active.plan.stage.name == "push_open" and active.action_index == 0:
                saw_push_effect = True
                assert backend.get_joint_angle("door__handle_hinge", 0) >= 0.12
                assert not bool(single_env.data.eq_active[latch_id])
                assert backend.is_object_grasped(
                    "arm", "door__door_handle"
                ).tolist() == [True]

        assert saw_pull_effect
        assert saw_push_effect
        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert [record.stage_name for record in runner.records] == [
            "pick_handle",
            "pull_handle",
            "push_open",
        ]
        assert backend.get_joint_angle("door__door_hinge", 0) >= 0.18
        assert backend.is_object_grasped("arm", "door__door_handle").tolist() == [True]
    finally:
        runner.close()
