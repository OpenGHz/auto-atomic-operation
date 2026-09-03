# PlaceGen rack-plate asset migration

This note records the canonical AAO copy of the PlaceGen draining-rack assets. It
supplements the [XML / Mesh / GS migration rules](xml_mesh_gs_migration_notes.md).

## Outcome

The migrated host scene is robot-less and self-contained apart from the already
canonical `plate2` mesh shared with the dishwasher task:

```text
assets/
├── meshes/
│   ├── rack_plate/rack-plate-0.obj
│   └── dishwasher_plate/plate2/plate2.obj  # shared canonical object mesh
└── xmls/scenes/rack_plate/
    └── demo.xml
```

`demo.xml` keeps the PlaceGen rack geometry, target support, plate pose, and five
fixed observation cameras. It uses relative `meshdir="../../../meshes"` paths so
the final scene can be loaded from any AAO checkout. The rack visual mesh is
render-only; nine invisible box ribs preserve the source collision representation
without turning the non-convex wire mesh into a solid collision hull. The plate
uses its source visual transform and a finite cylinder collision proxy.

The host contains no robot include, actuator, keyframe, or Gaussian asset. A robot
can be injected later as an ordered `SceneConfig` MJCF layer, for example with
`assets/xmls/robots/xf9600_mocap.xml`; this keeps robot definitions and scene assets
under separate ownership boundaries.

## P7 V4 placement task

The runnable `rack_plate_p7_v4_umi_v3` task composes this host with
`assets/xmls/robots/p7_arm_v4_with_umi_gripper_v3.xml` and the existing analytical
P7 IK binding:

```bash
aao-info rack_plate_p7_v4_umi_v3
aao-demo --config-name rack_plate_p7_v4_umi_v3
```

The scene places the draining rack on the table plane and keeps the plate upright
in a small geometric stand outside the rack. The stand's base, rear stop, and two
low side rails are physical support geometry; its low front guide is visual-only so
the gripper can leave the stand toward the rack. The source stand is deliberately
separated from the first rack rib, preventing the free plate from rolling into the
rack before the pick stage starts.

The task's pick stage raises the open gripper above the plate, translates to the
plate centre, and then descends between its faces. After a verified two-sided grasp,
the place stage first moves the held-object frame above the selected slot and then
lowers it into `rack_target_site`. Its `axis_alignment` goal constrains the plate's
physical normal while leaving its in-plane twist free. The final opening lets the
plate settle on the rack floor; the regression checks that no plate--rib or
robot--stand contact occurs while allowing the intentional base/floor support
contacts.

The focused end-to-end check is:

```bash
/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python scripts/run_tests_safe.py \
  --test-targets tests/test_rack_plate_p7_v4_umi_v3.py --max-concurrency=1
```

## Deliberate migration boundary

This migration copies scene assets only. It does not migrate:

- the PlaceGen gantry robot, spherical wrist, floating-hand proxies, or parallel jaw;
- PlaceGen-specific configuration, planners, perception pipelines, datasets, or checkpoints;
- source keyframes and actuators, whose layouts are robot-specific;
- robot or background Gaussian splats.

The original PlaceGen files remain source references. After this migration, maintain
the AAO copies and do not bounce edits between the two repositories.

## Provenance and integrity

The rack mesh was copied byte-for-byte from
`robust-placer/third_party/placegen/src/placegen/resources/mujoco/meshes/rack-plate-0.obj`.
The source path was last changed in PlaceGen commit
`93ed05dc04e8632939ca0b7123846f4b22dc0a18` (the later collision-proxy scene change
is tracked separately in commit `6711c836ab5dbd21b9d2a647d3b4739d10c812ef`). The
five-camera layout follows the fixed RGB-D variant
`gantry_rack_plate_rgbd_esdf.xml`; those cameras are scene observation metadata,
not robot assets. The canonical AAO files have these SHA-256 digests:

| Canonical file | SHA-256 |
|---|---|
| `assets/meshes/rack_plate/rack-plate-0.obj` | `863892cdc8116e632d02b37860c354817aa59bb136d0dc81d2d991fe0ef0fda4` |
| `assets/meshes/dishwasher_plate/plate2/plate2.obj` | `960f4113d5a9e6b123b836026f04889c45e30429ae4dda6bfc564f68e5757f93` |

`plate2.obj` is not duplicated: it is the existing canonical AAO payload from the
Dishwasher031 migration and is referenced by the new host scene. The source asset
bundle did not include redistribution terms; verify upstream terms before publishing
a release containing these third-party mesh bytes.

## Validation

Load the final top-level host scene rather than opening the source mesh or an
isolated XML fragment:

```bash
python - <<'PY'
from pathlib import Path
import mujoco

scene = Path("assets/xmls/scenes/rack_plate/demo.xml")
model = mujoco.MjModel.from_xml_path(str(scene))
print("loaded ok", "nq=", model.nq, "nu=", model.nu, "ncam=", model.ncam)
PY
```

The focused regression checks exact payload hashes, relative path resolution,
robot-less host structure, visual/collision separation, rack rib count, and the
optional host-plus-robot composition path:

```bash
python -m pytest tests/test_rack_plate_scene.py -q
```
