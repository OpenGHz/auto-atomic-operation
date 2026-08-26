"""Backend-neutral outputs produced by scene module adapters."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .config import AssetAssemblyLayerConfig


@dataclass(frozen=True)
class SceneContribution:
    """One self-contained contribution to a host scene.

    The host composer owns merging and temporary-file lifecycle.  An adapter
    only returns a fragment, semantic exports, and the files it consumed.
    """

    fragment: ET.Element
    semantic_refs: Mapping[str, str] = field(default_factory=dict)
    dependencies: tuple[Path, ...] = ()
    adapter: str = ""
    diagnostics: tuple[str, ...] = ()

    @property
    def digest(self) -> str:
        """Stable digest of fragment text and dependency metadata."""
        digest = hashlib.sha256()
        digest.update(ET.tostring(self.fragment, encoding="utf-8"))
        for path in sorted(self.dependencies, key=lambda item: str(item)):
            digest.update(str(path).encode("utf-8"))
            try:
                digest.update(hashlib.sha256(path.read_bytes()).digest())
            except OSError:
                digest.update(b"<missing>")
        return digest.hexdigest()


@dataclass(frozen=True)
class SceneArtifact:
    """Materialized scene plus semantic exports and provenance."""

    xml: str
    semantic_refs: Mapping[str, str] = field(default_factory=dict)
    dependencies: tuple[Path, ...] = ()
    digest: str = ""
    diagnostics: tuple[str, ...] = ()


class SceneAssembler(ABC):
    """Runtime adapter seam for one declarative asset-assembly layer.

    Configuration remains plain data; implementations are created by the
    runtime registry and receive a validated layer recipe for each compile.
    """

    @abstractmethod
    def assemble(self, config: AssetAssemblyLayerConfig) -> SceneContribution:
        """Resolve and compile ``config`` without mutating the host scene."""


__all__ = ["SceneArtifact", "SceneAssembler", "SceneContribution"]
