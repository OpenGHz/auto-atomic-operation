"""Backend-neutral scene composition contracts and the MuJoCo compiler.

Scene composition is deliberately split into two layers:

* :mod:`auto_atom.scene_composition.config` contains serialisable, validated
  recipes.  It has no simulator or vendor imports and is safe to round-trip
  through Hydra/Pydantic.
* :mod:`auto_atom.scene_composition.composer` resolves those recipes into a
  backend scene.  Adapters are registered in the implementation layer rather
  than encoded as live objects in configuration.

The package currently ships an MJCF renderer and the first two scene module
adapters (pre-authored MJCF and the normalized UniDoor package).  Adding a
different asset family does not require changing ``EnvConfig`` or the viewer.
"""

from .adapters import register_asset_adapter, register_scene_assembler
from .composer import (
    SceneComposer,
    compile_scene,
    compose_scene,
    load_composed_scene,
    materialize_scene,
)
from .config import (
    AssetAssemblyLayerConfig,
    MjcfLayerConfig,
    SceneConfig,
    SceneLayerConfig,
    TransformConfig,
)
from .contracts import SceneArtifact, SceneAssembler, SceneContribution
from .package import (
    PackageFrameConfig,
    PackagePayloadConfig,
    PackageValidationReport,
    SceneAssetArtifactDescriptor,
    SceneAssetComponentManifest,
    SceneAssetPackageDescriptor,
    load_component_manifest,
    load_package_descriptor,
    validate_package_payload,
)

__all__ = [
    "AssetAssemblyLayerConfig",
    "MjcfLayerConfig",
    "SceneConfig",
    "SceneLayerConfig",
    "TransformConfig",
    "SceneArtifact",
    "SceneAssembler",
    "SceneContribution",
    "SceneComposer",
    "compile_scene",
    "compose_scene",
    "load_composed_scene",
    "materialize_scene",
    "register_asset_adapter",
    "register_scene_assembler",
    "PackageFrameConfig",
    "PackagePayloadConfig",
    "PackageValidationReport",
    "SceneAssetArtifactDescriptor",
    "SceneAssetComponentManifest",
    "SceneAssetPackageDescriptor",
    "load_component_manifest",
    "load_package_descriptor",
    "validate_package_payload",
]
