"""Runtime adapter registry for declarative scene layers."""

from __future__ import annotations

from collections.abc import Callable

from ..config import AssetAssemblyLayerConfig
from ..contracts import SceneAssembler, SceneContribution

AssetAdapter = Callable[[AssetAssemblyLayerConfig], SceneContribution]
_ASSET_ADAPTERS: dict[str, AssetAdapter] = {}


def register_asset_adapter(
    adapter_id: str, adapter: AssetAdapter | SceneAssembler
) -> None:
    """Register a runtime adapter implementation for an ``adapter@version`` id."""

    normalized = str(adapter_id).strip()
    if not normalized or "@" not in normalized:
        raise ValueError("adapter_id must include an explicit @version")
    if normalized in _ASSET_ADAPTERS:
        raise ValueError(f"asset adapter already registered: {normalized}")
    implementation = (
        adapter.assemble if isinstance(adapter, SceneAssembler) else adapter
    )
    _ASSET_ADAPTERS[normalized] = implementation


register_scene_assembler = register_asset_adapter


def compile_asset_layer(layer: AssetAssemblyLayerConfig) -> SceneContribution:
    """Compile one asset layer through its registered adapter.

    Imports are lazy so importing the generic scene contract never imports
    optional vendor packages or MuJoCo-specific asset code.
    """

    adapter = _ASSET_ADAPTERS.get(layer.adapter)
    if adapter is None and layer.adapter == "unidoor.lever_door@1":
        # Built-ins are imported lazily, keeping the generic contract free of
        # vendor imports while still making the standard adapter available.
        from .unidoor import UniDoorSceneAssembler

        register_asset_adapter("unidoor.lever_door@1", UniDoorSceneAssembler().assemble)
        adapter = _ASSET_ADAPTERS.get(layer.adapter)
    if adapter is None:
        raise ValueError(f"unknown scene asset adapter: {layer.adapter!r}")
    return adapter(layer)


__all__ = [
    "compile_asset_layer",
    "register_asset_adapter",
    "register_scene_assembler",
]
