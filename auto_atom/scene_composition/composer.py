"""Ordered scene compiler and MuJoCo materialization boundary."""

from __future__ import annotations

import hashlib
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from .adapters import compile_asset_layer
from .adapters.mjcf import load_mjcf_fragment
from .config import AssetAssemblyLayerConfig, MjcfLayerConfig, SceneConfig
from .contracts import SceneArtifact, SceneContribution

_APPEND_SECTIONS = {
    "asset",
    "default",
    "worldbody",
    "equality",
    "tendon",
    "contact",
    "deformable",
    "actuator",
    "sensor",
    "tuple",
    "keyframe",
    "custom",
    "extension",
}
_SINGLETON_SECTIONS = {"compiler", "option", "size", "statistic", "visual"}


class SceneComposer:
    """Facade for the ordered scene-composition compiler.

    The stateless facade is useful to callers that want to inject/cache a
    compiler object, while the module-level functions remain convenient for
    one-shot environment construction.
    """

    def compile(self, config: SceneConfig) -> SceneArtifact:
        """Compile ``config`` into an inspectable artifact."""

        return compile_scene(config)

    def compose(self, config: SceneConfig) -> str:
        """Return the composed XML text."""

        return self.compile(config).xml

    def load(self, config: SceneConfig):
        """Compile and load a MuJoCo model."""

        return load_composed_scene(config)


def compile_scene(config: SceneConfig) -> SceneArtifact:
    """Compile a declarative scene into a deterministic XML artifact."""

    base = config.base.expanduser().resolve()
    if not base.is_file():
        raise FileNotFoundError(f"base scene XML not found: {base}")
    root = ET.parse(base).getroot()
    if root.tag != "mujoco":
        raise ValueError(f"base scene root must be <mujoco>: {base}")

    exports: dict[str, str] = {}
    dependencies: set[Path] = {base}
    diagnostics: list[str] = []
    for layer in config.layers:
        if isinstance(layer, MjcfLayerConfig):
            fragment, layer_dependencies = load_mjcf_fragment(layer)
            _merge_fragment(root, fragment)
            dependencies.update(layer_dependencies)
            continue

        contribution = compile_asset_layer(layer)
        _merge_fragment(root, contribution.fragment)
        dependencies.update(contribution.dependencies)
        diagnostics.extend(contribution.diagnostics)
        for key, value in contribution.semantic_refs.items():
            if _find_any_named(root, value) is None:
                raise ValueError(
                    f"semantic export {key!r} points to missing MJCF name {value!r}"
                )
            qualified_key = f"{layer.namespace}.{key}"
            if qualified_key in exports:
                raise ValueError(f"duplicate semantic export: {qualified_key}")
            exports[qualified_key] = value
            # A single-module scene can use the concise key.  Multiple modules
            # must use the namespace-qualified key to avoid ambiguity.
            if key not in exports:
                exports[key] = value

    xml = ET.tostring(root, encoding="unicode")
    digest = _digest(xml, dependencies)
    return SceneArtifact(
        xml=xml,
        semantic_refs=exports,
        dependencies=tuple(sorted(dependencies)),
        digest=digest,
        diagnostics=tuple(diagnostics),
    )


def compose_scene(config: SceneConfig) -> str:
    """Return the composed XML for inspection or downstream tooling."""

    return compile_scene(config).xml


@contextmanager
def materialize_scene(
    config: SceneConfig, artifact: SceneArtifact | None = None
) -> Iterator[Path]:
    """Materialize a compiled scene beside its host and clean it on exit."""

    artifact = artifact or compile_scene(config)
    base = config.base.expanduser().resolve()
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(base.parent),
        prefix="._aao_scene_",
        suffix=".xml",
        delete=False,
    ) as stream:
        stream.write(artifact.xml)
        path = Path(stream.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def load_composed_scene(config: SceneConfig, artifact: SceneArtifact | None = None):
    """Compile and load a scene through the single MuJoCo entry point."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("MuJoCo is required to load a composed scene") from exc

    if not config.layers and artifact is None:
        return mujoco.MjModel.from_xml_path(str(config.base.expanduser().resolve()))
    with materialize_scene(config, artifact) as path:
        return mujoco.MjModel.from_xml_path(str(path))


def _merge_fragment(host: ET.Element, fragment: ET.Element) -> None:
    if host.find("worldbody") is None:
        raise ValueError("base scene is missing <worldbody>")

    for child in list(fragment):
        tag = child.tag
        if tag in _SINGLETON_SECTIONS:
            _merge_singleton(host, child)
        elif tag in _APPEND_SECTIONS:
            _append_section(host, child)
        else:
            # Preserve future MJCF top-level sections rather than silently
            # dropping them.  A repeated named section is still rejected.
            _append_section(host, child)


def _merge_singleton(host: ET.Element, source: ET.Element) -> None:
    target = host.find(source.tag)
    if target is None:
        host.append(deepcopy(source))
        return
    # Host settings are authoritative.  Fill only attributes that the host
    # did not specify; this makes robot modules composable without allowing a
    # hidden module to rewrite global gravity/timestep.
    for key, value in source.attrib.items():
        if key not in target.attrib:
            target.set(key, value)
    for child in list(source):
        if child.tag == "default":
            _append_named_child(target, child, class_attribute=True)
        elif child.tag not in {item.tag for item in list(target)}:
            target.append(deepcopy(child))


def _append_section(host: ET.Element, source: ET.Element) -> None:
    target = host.find(source.tag)
    if target is None:
        target = ET.SubElement(host, source.tag, dict(source.attrib))
    for child in list(source):
        if child.tag == "body" and source.tag == "worldbody":
            _check_named_collisions(host, child)
        elif child.get("name") or child.get("class"):
            _check_named_collisions(host, child)
        target.append(deepcopy(child))


def _append_named_child(
    target: ET.Element, child: ET.Element, *, class_attribute: bool = False
) -> None:
    key = "class" if class_attribute else "name"
    value = child.get(key)
    if value and any(item.get(key) == value for item in target):
        raise ValueError(f"duplicate MJCF {key}={value!r} in <{target.tag}>")
    target.append(deepcopy(child))


def _check_named_collisions(host: ET.Element, subtree: ET.Element) -> None:
    for element in subtree.iter():
        name = element.get("name")
        if name and _find_named(host, element.tag, name) is not None:
            raise ValueError(f"base scene already defines {element.tag}/{name}")
        if element.tag == "default":
            cls = element.get("class")
            if cls and any(item.get("class") == cls for item in host.iter("default")):
                raise ValueError(f"base scene already defines default class {cls}")


def _find_named(root: ET.Element, tag: str, name: str) -> ET.Element | None:
    for element in root.iter(tag):
        if element.get("name") == name:
            return element
    return None


def _find_any_named(root: ET.Element, name: str) -> ET.Element | None:
    for element in root.iter():
        if element.get("name") == name:
            return element
    return None


def _digest(xml: str, dependencies: set[Path]) -> str:
    digest = hashlib.sha256(xml.encode("utf-8"))
    for path in sorted(dependencies, key=str):
        digest.update(str(path).encode("utf-8"))
        try:
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()


__all__ = [
    "SceneComposer",
    "compile_scene",
    "compose_scene",
    "load_composed_scene",
    "materialize_scene",
]
