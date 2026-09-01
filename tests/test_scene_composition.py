"""Contract tests for the generic scene-composition seam."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from auto_atom.scene_composition import (
    AssetAnchorConfig,
    AssetAssemblyLayerConfig,
    AssetScaleRuleConfig,
    MjcfLayerConfig,
    SceneConfig,
    SceneContribution,
    SceneComposer,
    SceneAssetPackageDescriptor,
    TransformConfig,
    compose_scene,
    materialize_scene,
    register_asset_adapter,
    load_package_descriptor,
    load_component_manifest,
    validate_package_payload,
    apply_asset_normalization,
)
from auto_atom.scene_composition.migrate import migrate_legacy_catalog


def _write_scene(path: Path) -> None:
    path.write_text(
        """<mujoco model="host">
  <option timestep="0.01"/>
  <worldbody><body name="host_body"/></worldbody>
</mujoco>""",
        encoding="utf-8",
    )


def _write_layer(path: Path, name: str) -> None:
    path.write_text(
        f"""<mujoco model="{name}">
  <compiler angle="radian"/>
  <asset><mesh name="{name}_mesh" file="{name}.obj"/></asset>
  <worldbody><body name="{name}_body"><geom name="{name}_geom" type="box" size=".01 .01 .01"/></body></worldbody>
  <contact><exclude body1="host_body" body2="{name}_body"/></contact>
  <equality><weld body1="host_body" body2="{name}_body"/></equality>
</mujoco>""",
        encoding="utf-8",
    )
    (path.parent / f"{name}.obj").write_text(
        "v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n",
        encoding="utf-8",
    )


def test_scene_config_is_discriminated_and_round_trips() -> None:
    config = SceneConfig.model_validate(
        {
            "base": "host.xml",
            "layers": [
                {"kind": "mjcf", "path": "robot.xml"},
                {
                    "kind": "asset_assembly",
                    "package": "package.json",
                    "adapter": "example@1",
                    "selection": {"part": "P1"},
                    "namespace": "part",
                    "placement": {
                        "position": [1, 2, 3],
                        "orientation_xyzw": [0, 0, 0, 1],
                    },
                },
            ],
        }
    )
    restored = SceneConfig.model_validate(config.model_dump(mode="json"))
    assert restored == config
    assert isinstance(restored.layers[0], MjcfLayerConfig)
    assert isinstance(restored.layers[1], AssetAssemblyLayerConfig)
    assert TransformConfig(position=[1, 2, 3]).position == (1.0, 2.0, 3.0)


def test_generic_asset_normalization_scales_meshes_geometry_and_anchor() -> None:
    fragment = ET.fromstring(
        """<mujoco>
          <asset><mesh name="panel_mesh" scale="1 1 1"/></asset>
          <worldbody>
            <body name="panel" pos="0 0 0">
              <geom name="panel_collision" type="box" pos="0.5 0 0.5" size="0.5 0.02 0.5"/>
              <body name="handle" pos="0.8 0 0.5">
                <site name="grasp" pos="0.1 0 0" size="0.01"/>
              </body>
            </body>
          </worldbody>
        </mujoco>"""
    )
    diagnostics = apply_asset_normalization(
        fragment,
        scaling=(
            AssetScaleRuleConfig(
                bodies=("panel",),
                preserve_bodies=("handle",),
                meshes=("panel_mesh",),
                source_bounds="panel.bounds",
                axis="z",
                target_extent_m=2.0,
            ),
        ),
        anchors=(
            AssetAnchorConfig(
                bodies=("handle",),
                source_bounds="panel.bounds",
                coordinates={
                    "x": {"edge": "max", "offset_m": 0.08},
                    "z": {"value_m": 1.0},
                },
            ),
        ),
        metadata={"panel": {"bounds": [[0.0, -0.02, 0.0], [1.0, 0.02, 1.0]]}},
    )
    mesh = fragment.find("asset/mesh")
    assert mesh is not None
    assert mesh.get("scale") == "2 2 2"
    collision = fragment.find(".//geom")
    assert collision is not None
    assert collision.get("pos") == "1 0 1"
    assert collision.get("size") == "1 0.04 1"
    handle = next(
        body for body in fragment.iter("body") if body.get("name") == "handle"
    )
    assert handle.get("pos") == "2.08 0 1"
    site = fragment.find(".//site")
    assert site is not None
    assert site.get("pos") == "0.1 0 0"
    assert diagnostics == (
        "scaled panel.bounds to 2m (factor=2)",
        "positioned anchor bodies: handle",
    )


def test_fixed_asset_anchor_does_not_require_source_bounds() -> None:
    anchor = AssetAnchorConfig(
        bodies=("tool",),
        coordinates={
            "x": {"value_m": 0.1},
            "y": {"value_m": -0.2},
            "z": {"value_m": 0.3},
        },
    )
    assert anchor.source_bounds is None

    with pytest.raises(ValueError, match="source_bounds is required"):
        AssetAnchorConfig(
            bodies=("tool",),
            coordinates={"x": {"edge": "max"}},
        )


def test_mjcf_layer_preserves_nontrivial_top_level_sections(tmp_path: Path) -> None:
    host = tmp_path / "host.xml"
    layer = tmp_path / "layer.xml"
    _write_scene(host)
    _write_layer(layer, "robot")

    artifact = SceneComposer().compile(
        SceneConfig(base=host, layers=(MjcfLayerConfig(path=layer),))
    )
    root = ET.fromstring(artifact.xml)
    assert root.find("contact/exclude") is not None
    assert root.find("equality/weld") is not None
    # The host's singleton option remains authoritative.
    assert root.find("option").get("timestep") == "0.01"
    assert artifact.digest
    assert layer.resolve() in artifact.dependencies


def test_multiple_registered_asset_adapters_have_no_core_branch(tmp_path: Path) -> None:
    host = tmp_path / "host.xml"
    _write_scene(host)

    def fake_adapter(config: AssetAssemblyLayerConfig) -> SceneContribution:
        root = ET.Element("mujoco")
        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(
            worldbody,
            "body",
            {"name": f"{config.namespace}__fake_body"},
        )
        return SceneContribution(
            fragment=root,
            semantic_refs={"fake.body": f"{config.namespace}__fake_body"},
            adapter=config.adapter,
        )

    register_asset_adapter("example@1", fake_adapter)
    register_asset_adapter("example@2", fake_adapter)
    config = SceneConfig(
        base=host,
        layers=(
            AssetAssemblyLayerConfig(
                package=tmp_path,
                adapter="example@1",
                selection={"part": "P1"},
                namespace="one",
            ),
            AssetAssemblyLayerConfig(
                package=tmp_path,
                adapter="example@2",
                selection={"part": "P2"},
                namespace="two",
            ),
        ),
    )
    artifact = SceneComposer().compile(config)
    assert artifact.semantic_refs["one.fake.body"] == "one__fake_body"
    assert artifact.semantic_refs["two.fake.body"] == "two__fake_body"
    assert "one__fake_body" in artifact.xml and "two__fake_body" in artifact.xml


def test_duplicate_asset_namespaces_fail_before_adapter_execution(
    tmp_path: Path,
) -> None:
    _write_scene(tmp_path / "host.xml")
    with pytest.raises(ValueError, match="namespaces must be unique"):
        SceneConfig(
            base=tmp_path / "host.xml",
            layers=(
                AssetAssemblyLayerConfig(
                    package=tmp_path,
                    adapter="example@1",
                    selection={"part": "P1"},
                    namespace="same",
                ),
                AssetAssemblyLayerConfig(
                    package=tmp_path,
                    adapter="example@1",
                    selection={"part": "P2"},
                    namespace="same",
                ),
            ),
        )


def test_package_descriptor_rejects_legacy_catalog_field() -> None:
    with pytest.raises(ValueError, match="legacy_catalog"):
        SceneAssetPackageDescriptor.model_validate(
            {
                "schema": "aao.scene-asset-package/v1",
                "package_id": "example",
                "revision": "1",
                "payload_root": {"uri": "payload"},
                "units": {"length": "m", "angle": "rad", "mass": "kg"},
                "canonical_frame": {
                    "name": "package_local",
                    "up": "+z",
                    "handedness": "right",
                    "quaternion": "xyzw",
                    "transform_baked": True,
                },
                "components": {},
                "assembly_templates": [],
                "legacy_catalog": "payload",
            }
        )


def test_materialize_scene_cleans_temporary_file(tmp_path: Path) -> None:
    host = tmp_path / "host.xml"
    _write_scene(host)
    config = SceneConfig(base=host)
    with materialize_scene(config) as path:
        assert path.is_file()
        assert path.parent == host.parent
    assert not path.exists()


def test_batched_environment_compiles_one_artifact_for_replicas(
    tmp_path: Path, monkeypatch
) -> None:
    host = tmp_path / "host.xml"
    layer = tmp_path / "layer.xml"
    _write_scene(host)
    _write_layer(layer, "robot")
    import auto_atom.basis.mjc.mujoco_env as mujoco_env_module

    original = mujoco_env_module.compile_scene
    calls = 0

    def counting_compile(config):
        nonlocal calls
        calls += 1
        return original(config)

    monkeypatch.setattr(mujoco_env_module, "compile_scene", counting_compile)
    from auto_atom.basis.mjc.mujoco_basis import EnvConfig
    from auto_atom.basis.mjc.mujoco_env import BatchedUnifiedMujocoEnv

    env = BatchedUnifiedMujocoEnv(
        EnvConfig.model_validate(
            {
                "scene": {
                    "base": str(host),
                    "layers": [{"kind": "mjcf", "path": str(layer)}],
                },
                "batch_size": 2,
            }
        )
    )
    try:
        assert calls == 1
        assert len(env.envs) == 2
        assert env.envs[0].scene_artifact is env.envs[1].scene_artifact
    finally:
        env.close()


def test_unidoor_adapter_uses_canonical_descriptor_when_payload_is_present() -> None:
    descriptor = Path(
        "assets/scene_assets/unidoor_lever_right_hinge/scene_asset_package.json"
    )
    host = Path("assets/xmls/scenes/open_door_unidoor/demo.xml")
    robot = Path("assets/xmls/robots/p7_arm_v3_with_umi_gripper_v3.xml")
    payload = Path(
        "third_party/unidoor_lever_catalog_pipeline_right_hinge/product_space.json"
    )
    if (
        not descriptor.is_file()
        or not host.is_file()
        or not robot.is_file()
        or not payload.is_file()
    ):
        pytest.skip("tracked scene or robot assets are unavailable")
    pytest.importorskip("mujoco")
    config = SceneConfig(
        base=host,
        layers=(
            AssetAssemblyLayerConfig(
                package=descriptor,
                adapter="unidoor.lever_door@1",
                selection={"door": "D001", "handle": "H003"},
                namespace="door",
            ),
            MjcfLayerConfig(path=robot),
        ),
    )
    artifact = SceneComposer().compile(config)
    assert artifact.semantic_refs["door.door.hinge.joint"] == "door__door_hinge"
    assert "door__handle_grasp_center" in artifact.xml
    assert not any("legacy UniDoor" in message for message in artifact.diagnostics)


def test_unidoor_instances_can_share_a_host_with_distinct_namespaces() -> None:
    descriptor = Path(
        "assets/scene_assets/unidoor_lever_right_hinge/scene_asset_package.json"
    )
    host = Path("assets/xmls/scenes/open_door_unidoor/demo.xml")
    payload = Path(
        "third_party/unidoor_lever_catalog_pipeline_right_hinge/product_space.json"
    )
    if not descriptor.is_file() or not host.is_file() or not payload.is_file():
        pytest.skip("tracked scene assets are unavailable")
    config = SceneConfig(
        base=host,
        layers=tuple(
            AssetAssemblyLayerConfig(
                package=descriptor,
                adapter="unidoor.lever_door@1",
                selection={"door": door, "handle": handle},
                namespace=namespace,
            )
            for namespace, door, handle in (
                ("left", "D001", "H003"),
                ("right", "D002", "H004"),
            )
        ),
    )
    artifact = SceneComposer().compile(config)
    assert "left__door_hinge" in artifact.xml
    assert "right__door_hinge" in artifact.xml
    assert artifact.semantic_refs["left.door.hinge.joint"] == "left__door_hinge"
    assert artifact.semantic_refs["right.door.hinge.joint"] == "right__door_hinge"


def test_package_descriptor_validates_declared_payload_and_integrity_lock() -> None:
    descriptor = Path(
        "assets/scene_assets/unidoor_lever_right_hinge/scene_asset_package.json"
    )
    payload = Path(
        "third_party/unidoor_lever_catalog_pipeline_right_hinge/product_space.json"
    )
    if not descriptor.is_file() or not payload.is_file():
        pytest.skip("tracked package descriptor is unavailable")
    package = load_package_descriptor(descriptor)
    assert package.schema_id == "aao.scene-asset-package/v1"
    assert package.payload_root.uri.endswith(
        "third_party/unidoor_lever_catalog_pipeline_right_hinge"
    )
    report = validate_package_payload(descriptor)
    assert report.payload_root is not None
    assert report.warnings == ()
    assert report.selected_artifacts == 7


def test_legacy_catalog_migration_is_non_destructive_and_complete(
    tmp_path: Path,
) -> None:
    source = Path("third_party/unidoor_lever_catalog_pipeline_right_hinge")
    if not (source / "product_space.json").is_file():
        pytest.skip("external UniDoor payload is unavailable")
    report = migrate_legacy_catalog(source, tmp_path / "canonical")
    assert report.component_count == 102
    assert report.artifact_count == 278
    assert report.visual_artifact_count == 176
    assert report.collision_artifact_count == 102
    assert (report.output_root / "scene_asset_package.json").is_file()
    assert not (report.output_root / "components" / "D001" / "frame.obj").exists()
    migrated = validate_package_payload(report.output_root / "scene_asset_package.json")
    assert migrated.payload_root == source.resolve()
    manifest = load_component_manifest(report.output_root / "components" / "D001.json")
    assert manifest.schema_id == "aao.scene-asset-component/v1"
    assert len(manifest.artifacts["visual"]) == 2
    assert len(manifest.artifacts["collision"]) == 1
