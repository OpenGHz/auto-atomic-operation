# PlaceGen rack-plate asset migration

This note records the canonical AAO copy of the PlaceGen draining-rack assets. It
supplements the [XML / Mesh / GS migration rules](xml_mesh_gs_migration_notes.md).

## Outcome

The migrated host scene is robot-less and self-contained with the rack-local
`plate.obj` payload:

```text
assets/
├── meshes/
│   ├── rack_plate/rack-plate-0.obj
│   └── rack_plate/plate.obj                 # rack-plate object mesh
└── xmls/scenes/rack_plate/
    └── demo.xml
```

`demo.xml` keeps the PlaceGen rack geometry, target support, and one fixed
observation camera. It uses relative `meshdir="../../../meshes"` paths so
the final scene can be loaded from any AAO checkout. The rack visual mesh is
render-only; nine invisible box ribs preserve the source collision representation
without turning the non-convex wire mesh into a solid collision hull. The plate
uses its source visual transform and the same `plate.obj` mesh for physical
collision, so the rim and thickness seen by the simulator match the rendered
object.

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

The scene places the draining rack on the table plane and starts the smaller
175 mm plate with a 15 degree lean in a small geometric stand outside the rack.
The stand is intentionally minimal: one compact raised base (about 70 × 224 mm)
and four vertical box posts. The post centres are 28 mm apart across the plate
thickness and 180 mm apart along its diameter, leaving a narrow gap that catches
the real 20 mm-thick mesh instead of forcing it into an upright cylinder proxy.
The source stand is separated from the first rack rib, preventing the free plate
from rolling into the rack before the pick stage starts.

The task's pick stage uses two waypoints expressed in the plate's `object`
reference frame: it first moves above the plate front, then extends along the
same 25 mm shallow front offset for the two-sided grasp. The gripper orientation
uses an `axis_alignment` goal against the plate's object-local +X normal, so the
plate's lean is followed while its symmetric in-plane twist remains unconstrained.
This keeps the gripper base behind the plate while preserving a direct approach.
After a verified grasp,
the place stage moves the held-object frame above the selected slot and releases
17 mm above the settled centre. Its `axis_alignment` goal constrains the plate's
physical normal while leaving its in-plane twist free. The smaller plate then
settles on the rack floor; the regression forbids rack-rib contact during the
carried transfer and allows the intentional support contact that can occur after
release while the plate seats in the slot.

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
The single static front camera follows the fixed RGB-D variant
`gantry_rack_plate_rgbd_esdf.xml`; it is scene observation metadata, not a robot
asset. The robot-mounted wrist camera remains supplied by the P7 layer. The
canonical AAO files have these SHA-256 digests:

| Canonical file | SHA-256 |
|---|---|
| `assets/meshes/rack_plate/rack-plate-0.obj` | `246f635a77aa42b724186150f736205a908c6fae85768cf06d5e91ccad4bf74d` |
| `assets/meshes/rack_plate/plate.obj` | `1152b76cfc6d3dd8b876b05dd77e6ee9c98dff1dad77f9b08b850534062a1a4f` |

`plate.obj` is kept alongside the rack mesh because this task uses a smaller
175 mm diameter plate than the Dishwasher031 `plate2` asset. The source asset
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
