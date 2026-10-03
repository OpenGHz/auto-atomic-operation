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
frame, and the front `env1_cam` observation camera declared by
`aao_configs/scene/microwave_sweet_potato.yaml` (the embodiment adds its wrist
camera). The robot is injected as an ordered
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
position (±3 cm in x and y). The [randomization presets](#randomization-presets)
cover much wider ranges.

### Randomization presets

```bash
aao-demo task=microwave_sweet_potato randomization=microwave_sweet_potato/gravity
aao-demo task=microwave_sweet_potato randomization=microwave_sweet_potato/zero_gravity
```

Both presets move the microwave on the counter (x -10/+4 cm, y -3/0 cm, yaw ±15°;
the feet stay on the counter for every yaw) and place the tuber relative to it
(`reference: microwave`), so the tuber is always in front of the open cavity and
clear of the door. Both also open the door
to a random angle between 90° (square to the front) and the 152° stop, through
`task.randomization.joints` (`microwave_door_hinge: [-2.0943, -1.0123]`; the hinge
reads 0.5585 rad closed and opens toward negative values).

Below about 85° the door's free edge swings in front of the cavity. In door sweeps
at fixed angles (20 episodes per angle and preset), `gravity` succeeded 1/20 at
44°, 10/20 at 55° and 16/20 at 66°, and `zero_gravity` 2/20, 12/20 and 16/20; the
failures had the gripper or the carried tuber hitting the door and pushing it. At
78–84° a finger still brushed the door in single episodes and pushed it by up to
0.17 rad; one of 40 `gravity` episodes at 80° failed that way. At 88°, over 40
episodes per preset, nothing touched the door and the closest gripper mesh stayed
17.7 mm from its hulls. The two `gravity` failures at 88° fail the same way at 80°
and 84° without touching the door.

`gravity` slides the tuber on the counter (x -20/+4 cm, y +1/+5 cm, heading ±30°);
turning about the vertical keeps its settled resting pose. The gripper's home pose
is placed relative to the tuber (`reference: sweet_potato`), so it always starts
above and behind it, and then jittered by x ±6 cm, y -4/+5 cm, z -2/+8 cm and
±0.12/±0.12/±0.2 rad (roll/pitch/yaw). The grasp
tilts nose-down by 0–20° about the jaw axis, pivoting about the jaw contact. To get
that pivot, the pick is written in `sweet_potato_grasp_frame`, which is randomized
relative to the tuber with a roll offset (`reference: sweet_potato`,
`roll: [-0.349, 0]`).

- A jaw roll (tilt about the approach axis) does not fit here: the open jaws pass
  the resting tuber only 3 mm above the counter, and by the gripper geometry a 10°
  roll lowers the open lower finger by about 9 mm, into the counter.
- Fixing only the jaw axis (`axis_alignment`, as in `rack_plate`) and taking the
  pitch from the home pose also tilts the grasp. But the waypoints then do not turn
  with the tilt: at 15–20° the pads close about 4.5 mm higher on the tuber, and 8 of
  11 episodes tilted 15° or more dropped it in transit. With the rigid pivot, 2 of
  25 did.
- Beyond about 20° the pads bite a slanted section and the hanging tuber slides out;
  4 of 46 episodes tilted 20–30° dropped it. A tilt of 45° still inserts physically,
  but beyond about 30° the gripper body rises into the cavity lip; the render-only
  meshes clip the hulls by 7 mm at 35° and 28 mm at 45°.

`zero_gravity` sets `env.gravity: [0, 0, 0]`. The tuber floats 10–40 cm above its
rest height with a random heading (±30°), long-axis elevation (±20°) and spin about
its long axis (±180°), anywhere the front camera sees it. It need not be over the
counter: the proposal box (x -25/+45 cm, y -20/+5 cm around its rest position) is
wider than the counter top, and `visible_in` on `env1_cam` (bounding sphere, 8 px
margin) trims it to the view. The box's own limits only keep the tuber clear of
the counter top (z), the cabinet (+y) and the opened door (-x). Over 300 resets 93
tubers started beyond the counter's edges; the tuber mesh stayed at least 43 px
inside the 640×352 image, 8.6 cm above the counter top, and 9.0 cm and 11.8 cm
from the cabinet and door hulls.

`visible_in` names `env1_cam` instead of `all`: the gripper's home pose follows the
tuber and is sampled after it, so the wrist camera's view is not final when the
tuber is checked, and the executor rejects a check on an operator-mounted camera
in that order.

The pick fixes only the gripper's approach axis along the
tuber's long axis (`pick_orientation: approach_axis`) and lines up and closes on that
axis, so it does not depend on the tuber's spin. The jaws close at the gripper's own
roll, which comes from its home pose (±45°). The home pose follows only the tuber's
position (`reference: sweet_potato`, `follow: position`), so it always starts
10–20 cm above and 35–44 cm behind it, jittered by x ±6 cm and ±0.12/±0.2 rad of
pitch/yaw. With the default `follow: pose` it would orbit the
tuber's long axis with the spin: over 40 resets 15 started below the tuber and 19
upside down. A floating tuber is pushed by the closing jaws instead of resting on the
counter, which changes several settings:

- The EEF closes on the long axis instead of 9.3 mm above it, and up to 2 cm further
  back toward the blunt end (per-waypoint randomization of the closing position).
  Pads closing above the widest line, or on the tapered rear (the section narrows
  from 42 mm to 26 mm between 5 and 7 cm behind the centre), squeeze the free tuber
  forward out of the grip. With up to 3 cm back, 3 of 29 episodes lost the tuber
  from grasps 4.5–5.7 cm behind its centre.
- The cavity fits only about 10° of jaw roll. At 10–20° the linkage render meshes
  clip the cavity floor and ceiling by up to 6 mm, and from 20° the collision hulls
  touch. The place stage therefore levels the jaws in front of the cavity, which
  spins the held tuber about its long axis.
- The tuber's spin is random, so the place goal constrains only its long axis
  (`place_orientation: axis`). Without gravity the roll does not sag.
- A released tuber keeps its velocity; one traced release glided off the target at
  about 5 cm/s. The arm holds still for 15 updates before opening
  (`grasp.pre_release_settle_steps`): without the hold 9 of 20 episodes succeeded,
  and step clamping the final approach instead reached 16 of 20.
  The tuber is released 15 mm above the rest pose, clear of the dish, and the
  post-release settle is skipped.

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
the minimum gripper–microwave clearance is 5.4 mm. Each preset also runs two seeded
episodes headless and checks the camera set (`env1_cam` and the wrist camera), that
the gripper starts above the tuber, that a floating tuber's centre is inside the
front image, and the door range. In sweeps of the current presets (batch 1, 20
episodes per seed), the default randomization succeeded 20/20, and over four seeds
`zero_gravity` succeeded 80/80 and `gravity` 77/80. The three `gravity` failures
dropped the tuber while sliding it in after grasps tilted 15–16°, the occasional
slip of a tilted grasp described above. No episode showed gripper or tuber contact
with the door, or gripper contact with the cabinet or counter; the minimum
render-mesh clearances to the microwave hulls were 5.6 mm (`gravity`) and 0.6 mm
(`zero_gravity`).

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
