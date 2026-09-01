# UniDoor canonical package view

`scene_asset_package.json` is the tracked, relocatable package descriptor for
the UniDoor payload supplied under `third_party/`. It is a small schema view,
not a copy of the external payload. Catalog revision 1.6 includes the normalized
geometry identity, collision supplements, UV visual meshes, diffuse PNG atlases,
and source-material evidence. The runtime adapter resolves its declared
`payload_root`, validates selected manifest/artifact hashes, binds optional
visual materials, and emits namespaced semantic exports.

The adapter uses `product_space.json` as the private component index for this
package; pre-generated combination XMLs remain regression oracles. Components
without `outputs.<role>.visual.runtime` still use their geometry output without
a material, while texture-complete components select the declared runtime UV
mesh and texture bundle automatically.

The task config accepts `AAO_UNIDOOR_ASSET_PACKAGE`, which must point to a
canonical descriptor or a package directory containing
`scene_asset_package.json`. A raw catalog directory is intentionally rejected.

To generate those small sidecars for a deployment (without modifying the
source payload), run:

```bash
python -m auto_atom.scene_composition.migrate \
  third_party/unidoor_lever_catalog_pipeline_right_hinge \
  /path/to/canonical/unidoor --overwrite
```

The report is expected to contain 102 component manifests, 176 visual and 102
collision artifact references for the supplied 55×47 catalog.  The generated view keeps
the payload mounted separately and records legacy hashes/provenance explicitly.
