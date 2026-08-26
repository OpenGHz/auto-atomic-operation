# UniDoor canonical package view

`scene_asset_package.json` is the tracked, relocatable package descriptor for
the UniDoor payload supplied under `third_party/`. It is a small schema view,
not a copy of the 144 MiB OBJ bundle. The runtime adapter resolves its declared
`payload_root`, validates selected manifest/artifact hashes, and emits
namespaced semantic exports.

The adapter uses `product_space.json` as the private component index for this
package; pre-generated combination XMLs remain regression oracles. A future
package revision can replace that index with canonical component sidecars
without changing task configuration or copying mesh bytes.

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
