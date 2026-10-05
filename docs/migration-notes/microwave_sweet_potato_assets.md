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
embodiment.

There are no Gaussian assets for this scene and no host actuators or keyframes.

## Source boundary

The source bundle was delivered as `third_party/assets/microwave/` (a `microwave.xml`,
the visual `microwave_body.obj` / `microwave_door.obj`, and a `convex/` directory) and
`third_party/assets/sweet_potato.obj`. Both `third_party/` copies are ignored by git and
remain source references only; maintain the AAO copies. The source `microwave.xml` and
its `convex/*.xml` include fragments are not migrated; their structure is rewritten into
the two AAO includes with `microwave_`-prefixed names and classes.

The visual OBJs and the hulls follow the source split:

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
  body instead to turn the grasp about the jaw contact. A second such body,
  `sweet_potato_grasp_roll_frame`, sits on the same point turned 90° about +Z, so
  its x is the long axis; `zero_gravity` turns the jaws about the long axis with
  it (see below).
- `microwave_target` is a child body of `microwave` at local
  `(-0.0288, 0.0536, -0.0173)`, yaw +90°: the settled sweet-potato origin after a
  gentle release, with local +X along the rest long axis. It sits about 3 cm in front
  of the dish centre, so the tuber's front end rests about 2 mm behind the cabinet's
  front face (local y -0.034); centred on the dish, its front end would sit 4 cm deeper.
  With the door closed the tuber still clears it by 12 mm (4.5 mm at 4 cm out); the
  settled pose repeats within 1 mm. Its render-only `microwave_target_pad` gives the
  mask renderer a geom; it lies 2.5 cm behind the target, on the dish's flat glass.

## UMI v3 task

```bash
aao-info microwave_sweet_potato
aao-demo task=microwave_sweet_potato    # default embodiment: umi_v3_mocap
```

The jaw plane of `umi_gripper_v3` is 56 mm thick and opens to 100 mm. Only the two
finger meshes collide with the scene. A top-down grasp followed by a 90° roll would
stand the jaws vertically in a 100 mm cavity, so the default configuration and
`gravity` use one horizontal grasp throughout: EEF +X (approach) points along world
+Y into the cavity, and the jaws close along world X. `zero_gravity` grasps at any
jaw roll and places upright jaws only with a raised path and a partial opening
(see [Randomization presets](#randomization-presets)).

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
   tilt (`place_entry: square_up`; `zero_gravity` instead moves the held tuber there
   with its place goal, `place_entry: held`). Held-object goals in the target frame then line the tuber up 0.30 m in front
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

[Microwave Sweet Potato Randomization](../task-tuning/microwave_sweet_potato_randomization.md)
lists every randomized item of the task and both presets in one place.

Both presets move the microwave on the counter (x -10/+4 cm, y -3/0 cm, yaw ±15°;
the feet stay on the counter for every yaw) and place the tuber relative to it
(`reference: microwave`), so the tuber is always in front of the open cavity and
clear of the door. Both also open the door
to a random angle between 90° (square to the front) and the 152° stop, through
`task.randomization.joints` (`microwave_door_hinge: [-2.0943, -1.0123]`; the hinge
reads 0.5585 rad closed and opens toward negative values).

Both presets also randomize the cameras (`_cameras.yaml`). The front `env1_cam`
only turns about its fixed position: roll ±0.08 rad tilts the view, yaw
±0.15 rad mostly pans and pitch ±0.10 rad mostly turns the image in its plane
(the offsets add to its world roll/pitch/yaw). At every combination of these
extremes and of the microwave placement, the whole cabinet front, the cavity
target and the resting tuber's region stay at least 12 px inside the 640×352
frame; tilt is the tight axis, leaving the frame beyond about +6°/-9° alone. The
wrist camera's mounting offset moves in its own frame by x ±3 mm, y -1/+5 mm,
z -5/+2 mm (right, up, back) and ±5° about each axis. The lens sits just above
and in front of the gripper housing: moving it 3 mm down brings it within 1 cm
of the housing, 6 mm back hides the fingers behind it, and 6 mm sideways halves
one finger in the image. At all 64 corners of the chosen range both fingers keep
at least 53% of their default image area. A wrist camera samples this offset
directly, so the gripper's randomized home pose does not leak into it.

Below about 85° the door's free edge swings in front of the cavity. In door sweeps
at fixed angles (20 episodes per angle and preset):

- `gravity` succeeded 1/20 at 44°, 10/20 at 55° and 16/20 at 66°; the failures had
  the gripper or the carried tuber hitting the door and pushing it. At 78–84° a
  finger still brushed the door in single episodes and pushed it by up to
  0.17 rad; one of 40 episodes at 80° failed that way.
- `zero_gravity` succeeded 0/20, 0/20 and 18/20. Its place goal checks clearance,
  so 39 of the 42 failures stopped at planning: no place orientation kept 10 mm
  from the cabinet and door. In the other three the gripper pushed the door, by up
  to 0.32 rad.
- At 88° nothing touched the door in 40 episodes per preset.

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
the counter top (z), the cabinet (+y) and the opened door (-x). Over 300 resets 91
tubers started beyond the counter's edges; the tuber mesh stayed at least 25 px
inside the 640×352 image, 8.8 cm above the counter top, and 8.3 cm and 11.2 cm
from the cabinet and door hulls.

`visible_in` names `env1_cam` instead of `all`: the gripper's home pose follows the
tuber and is sampled after it, so the wrist camera's view is not final when the
tuber is checked, and the executor rejects a check on an operator-mounted camera
in that order.

The pick is written in `sweet_potato_grasp_frame` with the gripper's whole
orientation fixed in it (`pick_orientation: level`), and two chained frames carry
the grasp's attitude relative to the tuber:

- `sweet_potato_grasp_roll_frame` follows the tuber's full pose, then its roll (the
  turn about its own x, the long axis) is set to an absolute ±90°
  (`reference: absolute_world` on that axis). The jaws therefore close at any angle
  from level to upright whatever the tuber's spin, and because the spin is uniform
  the jaw roll relative to the tuber covers the full circle. The jaws are never
  upside down, and with both place postures a grasp is never more than 15° from
  the nearer one.
- `sweet_potato_grasp_frame` follows the roll frame and tilts 0–20° nose-down about
  its own x, now the jaw axis (`roll: [-0.349, 0]`), through the jaw contact. The
  approach axis leaves the tuber's long axis by up to 20°, in a direction around the
  long axis that is uniform relative to the tuber.

A roll offset is the only rotation offset that turns about a frame's own axis (yaw
turns about world z, pitch about neither), hence two frames rather than one. The
tilt's direction relative to the gripper matters inside the cavity, where the
levelled tuber leaves side-by-side jaws pitched by the tilt (upright jaws turn
sideways instead). In a sweep with symmetric tilts about the tuber's lateral axis,
nose-up pitch clipped the top linkage render meshes into the cavity ceiling from
about 7° (by up to 27 mm), while nose-down pitch kept at least 19 mm to -19° and
clipped only beyond -30°, as under gravity. Tilts of 20–35° lost 3 of 60 tubers,
all while sliding into the cavity.

The home pose follows only the tuber's
position (`reference: sweet_potato`, `follow: position`), so it always starts
10–20 cm above and 20–29 cm behind it, jittered by x ±6 cm, ±0.785 rad of roll
about the approach axis and ±0.12/±0.2 rad of pitch/yaw. With the default
`follow: pose` it would orbit the
tuber's long axis with the spin: over 40 resets 15 started below the tuber and 19
upside down. A floating tuber is pushed by the closing jaws instead of resting on the
counter, which changes several settings:

- The EEF closes on the long axis instead of 9.3 mm above it, and up to 2 cm further
  back toward the blunt end (per-waypoint randomization of the closing position).
  Pads closing above the widest line, or on the tapered rear (the section narrows
  from 42 mm to 26 mm between 5 and 7 cm behind the centre), squeeze the free tuber
  forward out of the grip. With the grasp 2–3 cm back, 20 of 60 episodes failed,
  14 of them losing the tuber from the grip in transport or on the way in.
- The place goal fixes only what the task needs: the tuber's long axis level
  (`place_orientation: nearest`, a `nearest_feasible` goal), at any heading and
  spin. The jaws must be either side by side or one above the other (jaw axis
  within 30° of level or of vertical). Among those orientations the one nearest
  to how the tuber arrives wins, provided the gripper and tuber keep 10 mm from
  the cabinet and door along the slide-in, the release and the retreat.
  - It is solved once, 0.36 m in front of the cavity, and kept for the rest of
    the stage.
  - The margin covers the gripper's lag behind its command, several
    millimetres: with a 3 mm planning margin, fingers brushed the cabinet while
    opening and backing out.
  - Over the 120 validation episodes the place stage turned the tuber by a
    median of 22° (at most 66°).
  - See the [design](../design/nearest-feasible-place-orientation.md) for the
    probe behind these settings.
- The upright posture needs more room than the 126 mm cavity gives freely, so it
  has its own height and release opening.
  - The jaws span 145 mm along their axis fully open, so upright jaws open only to
    claw 0.008 (60 mm between the pads, against 43 mm across the held tuber),
    spanning 105 mm (`one_above_other_release`).
  - The place waypoints run 30 mm higher (`one_above_other_lift`).
  - Even then the 10 mm margin leaves about 1 mm of play, so the posture often
    does not fit, and the nearest level orientation is used instead; 11 of the
    120 validation episodes placed upright.
  - Lifts of 0–15 mm never fit, and opening to 0.009 (5 mm per side) let the
    fingers drag the tuber out on the way back.
- The placement check measures against the chosen posture's release pose with
  the same 25 mm tolerance. What varies is the distance to the nominal target:
  over the 120 validation episodes, level jaws left the tuber 10–19 mm above it,
  upright jaws 42–46 mm.
- A released tuber keeps its velocity, so the arm holds still for 15 updates
  before opening (`grasp.pre_release_settle_steps`). Without the hold none of 20
  episodes placed the tuber: in traced ones it was still moving at about 14 cm/s
  and 56 mm off the release pose when checked, against 0.2 cm/s and 14 mm with
  the hold. The tuber is released 15 mm above the rest pose, clear of the dish,
  and the post-release settle is skipped.

### UMI mocap weld

`umi_gripper_v3_mocap.xml` welds `umi_interface` to the mocap body with
`solref="0.10 1"`, the value `xf9600_mocap.xml` uses for the dishwasher task. Only
`umi_interface` is gravity-compensated, so the gripper links and payload hang on that
weld. With a 0.3 s time constant and the 0.2 kg tuber held about 4 cm ahead of the
pads, the EEF pitched about 4 cm off its command and the post-grasp lift timed out.
`task=pick_and_place embodiment=umi_v3_mocap` succeeds 2/2 on this weld.

Cartesian step clamping (`cartesian_max_linear_step`) is intentionally not used.
Each clamped command is re-based on the measured EEF, so any steady weld sag
compounds on long lateral moves. With a 1.5 cm step, the default task lost the
tuber right after the grasp in both of two episodes.

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
The recorded run completes in 127 control updates with a 4.8 mm settled error, and
the minimum gripper–microwave clearance is 15.6 mm. Each preset also runs two seeded
episodes headless and checks the camera set (`env1_cam` and the wrist camera), that
the gripper starts above the tuber, that a floating tuber's centre is inside the
front image, the door range, (`gravity`) that the tuber settles within 25 mm of
the target, and (`zero_gravity`) that the pick frame sits on the jaw contact tilted
at most 20°, that the planned place path keeps 10 mm from the microwave, and that
the tuber floats within 25 mm of the chosen posture's release pose.

In sweeps of 20 episodes per seed (batch 1), the default randomization succeeded
60/60 and `gravity` 57/60 over seeds 201–203, and `zero_gravity` 120/120 over seeds
201–206, 11 of them with upright jaws. The three `gravity` failures dropped the tuber
on the way into the cavity; the
[randomization summary](../task-tuning/microwave_sweet_potato_randomization.md#验证)
lists the commands that replay them. No episode touched the door, and the gripper never
touched the cabinet or counter; the closest gripper mesh stayed 15.6 mm (default),
15.9 mm (`gravity`) and 3.5 mm (`zero_gravity`) from the microwave hulls. The
target site stays at least 103 px inside the randomized front image.

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
