# Microwave and sweet-potato asset migration

This note records the canonical AAO copy of the microwave and sweet-potato assets and
the UMI v3 demo built on them. It supplements the general
[XML / Mesh / GS migration rules](xml_mesh_gs_migration_notes.md).

## Outcome

The migrated host scene is robot-less and self-contained:

```text
assets/
├── meshes/microwave_sweet_potato/
│   ├── microwave/
│   │   ├── microwave_body.obj                 # cabinet visual scan (517k vertices)
│   │   ├── microwave_door.obj                 # door visual scan (133k vertices)
│   │   └── convex/
│   │       ├── microwave_body/part_000..051.obj   # 52 cabinet convex hulls
│   │       └── microwave_door/part_000..063.obj   # 64 door convex hulls
│   └── sweet_potato/sweet_potato.obj          # 135k-vertex scan
└── xmls/scenes/microwave_sweet_potato/
    ├── demo.xml
    └── includes/
        ├── microwave_assets.xml               # visual + hull meshes, materials
        └── microwave_common.xml               # cabinet geoms, door body, hinge
```

`demo.xml` uses `meshdir="../../../meshes"`; no path is absolute. It contains the
counter, the fixed microwave, the free `sweet_potato`, the `microwave_target` placement
frame, and the `env0_cam` / `env1_cam` observation cameras declared by
`aao_configs/scene/microwave_sweet_potato.yaml`. The robot is injected as an ordered
MJCF layer (`assets/xmls/robots/umi_gripper_v3_mocap.xml`) by the `umi_v3_mocap`
embodiment (formerly `basis_mocap_eef_umi_v3`).

There are no Gaussian assets for this scene and no host actuators or keyframes.

## Source boundary

The source bundle was delivered as `third_party/assets/microwave/` (a `microwave.xml`,
the visual `microwave_body.obj` / `microwave_door.obj`, and a `convex/` directory) and
`third_party/assets/sweet_potato.obj`. Both `third_party/` copies are ignored by git and
remain source references only; maintain the AAO copies. The source `microwave.xml` and
its `convex/*.xml` include fragments are not migrated; their structure is rewritten into
the two AAO includes with `microwave_`-prefixed names and classes.

The two visual OBJs arrived after the first migration pass, which had rendered the
hulls directly. They now follow the source split:

- `microwave_body_visual` and `microwave_door_visual` use the source visual class:
  render-only (`contype="0" conaffinity="0"`), massless, group 0.
- The 116 hulls use the source collision class and are hidden in group 3 with the
  source's translucent green; enable group 3 in the viewer to inspect them.
- The visual scans carry only per-vertex colour (no UVs, MTL, or texture), and MuJoCo
  3.12 has no per-vertex colour channel. Each visual geom therefore uses one material
  set to the median vertex colour of its scan. Splitting the scans into colour-cluster
  sub-meshes was prototyped and rejected: it needs derived mesh files and posterises
  the door into visible patches, while the scan geometry already carries the look.

## Microwave

| Item | Canonical value |
|---|---|
| Frame | Source frame, metres, +Z up; door faces -Y and hinges on the -X side |
| Placement | `microwave` body at `(0.04, 0.13, 0.1874967)`; lowest foot hull is 0.1274967 m below the origin, so the feet rest on the 0.06 m counter |
| Cabinet | Fixed to the world (no joint); `microwave_body_visual` plus 52 hull geoms of class `microwave_hull` (source collision class: density 250, `friction="0.8 0.02 0.002"`, `condim=4`, group 3) |
| Door | `microwave_door` child body with `microwave_door_visual`, 64 hulls, and `microwave_door_hinge` (axis +Z at local `(-0.2241, -0.0421, 0)`, source damping/armature) |
| Contact exclude | cabinet ↔ door, as in the source, because hulls overlap at the hinge seam |

The source hinge coordinates are kept: `0` rad is the source-authored door pose (about
32° ajar) and the range is `[-120°, 32°]`, with `+32°` the closed stop. Like the
dishwasher migration, the placement-ready state is baked rather than stored in a
keyframe: the door body is authored at `-π/2` about the hinge and the joint has
`ref="-1.5707963"`, so a bare `MjData` starts with the door swung 122° open toward -X.
Setting the hinge to `0` returns the door body to the identity pose.

The hull convex decomposition models the turntable as a shallow glass dish (rim at
about local z = -0.031, centre near local `(-0.035, 0.08)`, radius about 0.12 m). The
cavity opening spans local x ∈ [-0.172, 0.094] and z ∈ [-0.055, 0.071].

## Sweet potato

- Body `sweet_potato`, freejoint `sweet_potato_joint`, origin site `sweet_potato_site`.
  The body frame is the scan frame: local +X is the 0.162 m long axis.
- The scan's enclosed volume is unreliable (19 cm³ against a 193 cm³ convex hull;
  20 small debris shells and self-overlap). The mesh asset therefore uses
  `inertia="convex"` and the collision geom sets `mass="0.2"` (hull volume at about
  1.05 g/cm³).
- One mesh asset serves both geoms: the visual geom renders the full scan, the
  collision geom uses its convex hull bounded by `maxhullvert="128"`.
- MuJoCo ignores OBJ vertex colours, so the skin material uses the median scan colour.
- The scan is not stable "local +Z up"; the authored pose is its simulated rest pose
  (about 31° of roll about the long axis), with the long axis pointing at the
  microwave (+Y). It drifts by less than 0.5 mm over one second.
- `sweet_potato_grasp_site` marks the jaw contact on the 45 mm section 40 mm behind
  the centre. Its orientation is world-aligned in the rest pose (+Y along the long
  axis toward the microwave, +Z up), so grasp waypoints written in it follow the
  tuber's pose. The static, geometry-less body `sweet_potato_grasp_frame` starts on
  the same pose. Sites cannot be randomized, so a randomization preset moves this
  body instead to turn the grasp about the jaw contact.
- `microwave_target` is a child body of `microwave` at local
  `(-0.030, 0.0856, -0.0165)`, yaw +90°: the settled sweet-potato origin after a gentle
  release, with local +X along the rest long axis. Its render-only
  `microwave_target_pad` gives the mask renderer a geom.

## UMI v3 task

```bash
aao-info microwave_sweet_potato
aao-demo task=microwave_sweet_potato    # default embodiment: umi_v3_mocap
```

The jaw plane of `umi_gripper_v3` is 56 mm thick and opens to 100 mm. Only the two
finger meshes collide with the scene. A top-down grasp followed by a 90° roll would
stand the jaws vertically in a 100 mm cavity, so the task uses one horizontal
grasp throughout. EEF +X (approach) points along world +Y into the cavity, and
the jaws close along world X.

1. `pick_sweet_potato`: approach from behind and above, drop to grasp height, and slide
   the open jaws around the blunt end. The waypoints are written in
   `microwave_sweet_potato.grasp_site` (`sweet_potato_grasp_site` by default); the
   EEF targets 24.1 mm ahead of and 9.3 mm above it. This puts the pads on the
   45 mm section 40 mm behind the centre: the contact band lies 4–16 mm below the
   jaw mid-plane, and the pads close about 24 mm behind `eef_pose`. Named parameters
   select the orientation goal (`pick_orientation`), the approach and grasp heights,
   and an optional per-waypoint randomization of the closing position.
2. `place_sweet_potato_in_microwave`: 0.36 m in front of the target, the gripper first
   turns its jaw axis horizontal and square to the cavity axis, keeping any grasp
   tilt. Held-object goals in the target frame then line the tuber up 0.30 m in front
   of the target, slide it in 22 mm above its rest height to clear the dish rim, and
   lower it to `microwave_sweet_potato.release_height` (6 mm) above the rest pose. By
   default every goal fixes the full settled orientation
   (`place_orientation: settled`): the long axis on target +X with the
   tuber's ~31° resting roll. An `axis_alignment` goal with free roll let the
   compliant weld's roll sag accumulate under the tuber's weight; in a traced
   failure the tuber was released about 16° off its resting roll and tipped over on
   the dish. After release and a
   30-update settle, the jaws back straight out.

The home freejoint pose keeps the gripper level behind the counter. The default
randomization jitters the home EEF pose (±5 cm x, ±4 cm y, -2/+4 cm z) and the tuber
position (±3 cm in x and y).

### UMI mocap weld

`umi_gripper_v3_mocap.xml` previously used a weld time constant of `0.3 s`. Only
`umi_interface` is gravity-compensated, so the gripper links and payload hang on that
weld. With the 0.2 kg tuber held about 4 cm ahead of the pads, the EEF pitched about
4 cm off its command and the post-grasp lift timed out. The weld now uses
`solref="0.10 1"`, the value `xf9600_mocap.xml` adopted for the dishwasher task.
`pick_and_place_umi_v3` (now `task=pick_and_place embodiment=umi_v3_mocap`) still
succeeds 2/2 with the change.

Cartesian step clamping (`cartesian_max_linear_step`) is intentionally not used.
Each clamped command is re-based on the measured EEF, so any steady weld sag
compounds on long lateral moves. With a 1.5 cm step, the first prototype sank
from the commanded 9 cm grasp height into the counter edge.

### Validation

```bash
python -m pytest tests/test_microwave_sweet_potato_scene.py \
  tests/test_microwave_sweet_potato_umi_v3.py -q
```

The scene test checks payload integrity, relative paths, robot-less host structure,
the visual/collision split, the door's baked open state, the sweet-potato rest pose and release
settling, and host-plus-UMI composition. The end-to-end test runs the seeded task
headless. It requires both finger pads on the tuber, no tuber–microwave contact
while carried, no finger–microwave contact, and positive clearance between every
gripper mesh (including render-only meshes) and the microwave hulls. The visual
scans sit about 2 mm inside the hulls, so hull clearance also bounds visual clipping. It also
requires a final error of at most 0.025 m after one more second of free settling.
The recorded run completes in 127 control updates with a 5.1 mm settled error, and
the minimum gripper–microwave clearance is 5.4 mm. In a 40-episode sweep of the default
randomization (batch 1, four seeds) the task succeeded 40/40.

## Provenance and integrity

The OBJ files are byte-identical to the delivered bundle; transforms live in the MJCF.

| Canonical file(s) | SHA-256 |
|---|---|
| `sweet_potato/sweet_potato.obj` | `7312cab781ab477f215944e4a4808505ca0656d3bec243b5ecd11820154a51b3` |
| `microwave/microwave_body.obj` | `8519021d027bfee55e0653d944ccc70830bf1e3a14d1181001f91e5a0eceba7b` |
| `microwave/microwave_door.obj` | `fa7df837a5035f1f46bea85c7e607cdd489936259127e4a0980187240c871323` |
| `microwave/convex/**/*.obj` (116 files, manifest digest) | `547c5caf6e93975d2196b84a28cdab10e79345c7eaec04af5cb75e723e8b9e69` |
| Source `microwave.xml` (not migrated; reference only) | `b344c0342febb2f812f5fc167f6d77fd196490d9199909034fd9373547975820` |

The manifest digest is the SHA-256 of the newline-joined lines
`"<file sha256>  <path relative to microwave/convex>"` in sorted path order, as
computed in `tests/test_microwave_sweet_potato_scene.py`. The bundle did not include
redistribution terms; verify upstream terms before publishing a release containing
these mesh bytes.
