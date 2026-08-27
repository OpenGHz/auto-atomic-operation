# Dishwasher031 and plate2 asset migration

This note records the canonical AAO copy of the dishwasher placement assets and the
boundary of this migration. It supplements the general
[XML / Mesh / GS migration rules](xml_mesh_gs_migration_notes.md).

## Outcome

The migrated asset family is self-contained under AAO and has no runtime dependency on
the original `robust-placer/third_party/assets` or PlaceGen checkout:

```text
assets/
├── meshes/dishwasher_plate/
│   ├── dishwasher031/
│   │   ├── Body001.obj
│   │   ├── button_lock.obj
│   │   ├── button_power.obj
│   │   ├── door.obj
│   │   ├── rack0.obj
│   │   ├── rack1.obj
│   │   └── T_BC001.png
│   └── plate2/plate2.obj
├── collision/dishwasher_plate/
│   ├── dishwasher031_rack1_wire_proxy.json
│   ├── dishwasher031_rack1_vertical_policy.json
│   ├── plate2_vertical_cylinder.json
│   └── plate2_vertical_cylinder_centers.npy
└── xmls/scenes/dishwasher_plate/
    ├── demo.xml
    └── includes/
        ├── dishwasher031_assets.xml
        ├── dishwasher031_common.xml
        ├── dishwasher031_rack1_articulation.xml
        ├── dishwasher031_actuators.xml
        └── dishwasher031_rack1_vertical_collision.xml
```

`demo.xml` is the canonical scene with a horizontal plate and consumes the generated
255-box rack-wire include. It is a robot-less host scene: a robot is added as an ordered
`MjcfLayerConfig`, just like other AAO scenes. Shared includes are the single source of truth
for dishwasher assets, the door/buttons/lower-rack mechanism, upper-rack articulation, and
all five actuators. There is no parallel horizontal-placement or static dishwasher scene.

## Deliberate migration boundary

This change migrates scene assets, not the source project's execution stack. In particular,
it does not migrate:

- the source gantry robot, its three actuators, or its keyframe;
- external automatic path generation or an independent penetration audit;
- a task operator for opening the door, moving a rack, or pressing a button;
- configuration-driven joint locking;
- unused source meshes (`Dishwasher031.obj`, `Door001.obj`, `Shelf001.obj`,
  `Shelf002.obj`, `Button001.obj`, and `Button002.obj`).

The subsequent object-centric motion-goal work adds the runnable
`aao_configs/dishwasher_plate.yaml`. It composes `basis_mocap_eef_xf9600`, selects this host
through `scene_name: dishwasher_plate`, and keeps the plate goal in object space: the plate's local
`+Z` axis is aligned with the rack target frame's `+Y` axis while twist about that normal is
left free. It does not rotate or otherwise rewrite `dishwasher_rack1_target_site`: the site
retains its authored identity orientation, and YAML independently selects that unchanged
frame's local `+Y` as the target axis. AAO executes the declared object waypoints directly;
this task does not claim automatic collision-free path generation.

## Articulated mechanism and default state

The source asset is articulated, but the placement task starts after appliance setup. The
canonical scene retains that articulation and expresses the placement-ready state through
joint `ref` values:

| Joint | Type and range | Default `qpos0` |
|---|---|---|
| `dishwasher_door_joint` | hinge, `[0, 0.6035987755982988] rad` | `0.6035987755982988` (open) |
| `dishwasher_button_lock_joint` | slide, `[0, 0.002] m` | `0` |
| `dishwasher_button_power_joint` | slide, `[0, 0.002] m` | `0` |
| `dishwasher_rack0_joint` | slide, `[0, 0.33] m` | `0` (retracted) |
| `dishwasher_rack1_joint` | slide, `[0, 0.33] m` | `0.33` (extracted) |

No host keyframe is needed. A bare `MjData` and a host composed with a robot layer both
start with the door open and upper rack extracted. Setting the door and upper-rack joint
positions to zero returns the door body to the closed identity transform and the rack body
to the retracted identity transform. The door visual, door-local collision proxies, two
button bodies, upper-rack collision proxies, and target site all move with those joints.

The host provides five source-equivalent effort motors with gears `50` for the door and
racks and `0.5` for the buttons. These are force/torque inputs, not position targets and not
joint locks. In a host-plus-Robotiq composition the five host actuators precede the gripper
actuator, so low-level `env.step()` calls must provide a full `model.nu` control vector in
compiled-model actuator order. Named operator APIs remain the safe interface; a one-element
"gripper action" must not be passed as a global control vector because it would address the
door motor instead.

Future configuration-driven locking should select joints by name and enforce their reset
positions with constraints. It should not emulate a lock with motor control or very large
gains. Until that contract exists, the placement setup relies on the authored damping,
friction, and zero motor control; it is approximately stationary but not mathematically
locked.

The upstream buttons used `springref=-1`, which drives a nominal `0..2 mm` button roughly
`6.6 mm` through its lower limit under zero control. A zero spring reference has the
opposite problem: gravity slowly self-presses both buttons to their upper limit while the
door is open. The canonical AAO asset therefore uses `springref=-0.0025 m`; with stiffness
`4 N/m` this gives a `10 mN` released-state preload, slightly above the compiled button
gravity load of about `8.83 mN`. This keeps both buttons at the released stop for a tested
10-second zero-control horizon. It is a documented dynamics correction, not a byte-for-byte
transcription of the faulty upstream parameter.

The plate remains the task's manipulated freejoint object. Its body/freejoint names are
`plate2` and `plate2_joint`, matching the AAO object-handler convention; the other five
joints belong to the dishwasher mechanism. The target is a child
`dishwasher_rack1_target` body with a
`dishwasher_rack1_target_site` site, so it follows rack motion and future task configuration
can reference a real pose-bearing entity rather than infer a pose from a visual marker.

## Collision representations

All visual meshes use `contype="0" conaffinity="0"`. This is essential for the wire rack:
allowing MuJoCo to convexify its visual OBJ would fill the slots with a non-existent solid
wall.

- The fixed cabinet uses explicit box proxies. The door uses its source-local oriented box,
  latch boxes, and cylindrical handle rather than open-pose world AABB slices; these
  proxies therefore remain correct while the door moves.
- The lower rack uses rack-local box proxies.
- The upper rack uses 255 rack-local, mesh-derived wire boxes. One selected longitudinal
  segment is split into front/contact/back parts without changing their union.
- `plate2_collision` is a finite cylinder enclosing the plate visual mesh.
- The copied 388-sphere cylinder cover is optional offline collision metadata; AAO's MuJoCo
  scene does not consume it implicitly.

## Provenance and integrity

The source bytes were found under the parent workspace's ignored
`third_party/assets/dishwasher/Dishwasher031` and `third_party/assets/plate` directories.
PlaceGen revision `4e52f7667c5aaa8009cb8c46152411eea981a507` contained byte-identical packaged copies and
the static task scenes used as the migration reference. After migration, the AAO copies are
the canonical maintenance location; changes should not be bounced back to the source tree.

| Canonical file | SHA-256 |
|---|---|
| `dishwasher031/Body001.obj` | `c108d7bac6641ad2d02c384137bbadefc6c35ff4cc22c63714141ed554c0934f` |
| `dishwasher031/button_lock.obj` | `7a16903533b6e65ff55688458857830003c21fb3855b5cfe5679c2401cec0201` |
| `dishwasher031/button_power.obj` | `278ee26ee2bcc96a8949e12e0e140e3f8afdc0a4312ded0a2dbe3e4a5e4c3192` |
| `dishwasher031/door.obj` | `71367e965b5a325e3537a4b4f9a73bdeaff621bf39ac58a9e4aa8dcbe44f5ce7` |
| `dishwasher031/rack0.obj` | `985c9ae19ecc54944c125ad56bea5cecb5441673847a4837340936725f2b1c6c` |
| `dishwasher031/rack1.obj` | `5204f9392cb5418b3e8ddf1920632619bdffe21f134a35cbcac5fc71f4e87dde` |
| `dishwasher031/T_BC001.png` | `78f413b3a878102a210d843eb4a7c0d024b99219064a1c991648b5ae397dfca0` |
| `plate2/plate2.obj` | `960f4113d5a9e6b123b836026f04889c45e30429ae4dda6bfc564f68e5757f93` |

The OBJ/PNG files are unchanged; transforms live in the MJCF. Units are metres and radians,
the scene is right-handed with +Z up, and MJCF quaternions use wxyz order.

The upstream asset bundle did not include provenance or redistribution terms. The AAO MIT
license therefore must not be interpreted as relicensing these third-party mesh/texture
bytes. Confirm the upstream terms before publishing or redistributing a release containing
them.

## Validation

The regression test validates exact payload hashes, the robot-less host, the
host-plus-Robotiq composition path, joint and motor contracts, default and retracted
mechanism transforms, moving-frame ownership, plate axis, visual collision flags, generated
wire count, a 10-second zero-control dynamics test, shared-definition ownership, and
collision-manifest bindings:

```bash
python -m pytest \
  tests/test_dishwasher_plate_scene.py -q
```

The runnable task composes successfully and is discoverable through the public task CLI:

```bash
aao-info dishwasher_plate --verbose --no-progress
aao-demo --config-name dishwasher_plate \
  +env.viewer.disable=true +max_updates=6000 +print_updates=false
```

The task uses `basis_mocap_eef_xf9600`. In the bounded actuator comparison, XF9600 was the
only existing gripper that passed the required held-object lift: its two-sided grasp closed
in 6 updates, and the direct 0.15 m lift completed in 46 updates with semantic object errors
of `0.01882 m` and `0.09819 rad`. Robotiq's wide cylinder contact slipped inside the gripper,
while UMI v3 still had `0.05304 m` / `0.24893 rad` EEF error after 1200 lift updates.

The current full physical run is nevertheless not recorded as an end-to-end success. After
the verified XF9600 lift, the first `axis_alignment` waypoint reaches its derived EEF command
but loses the two-sided plate grasp, so execution fails with
`controlled_frame_validation_failed` (`expected 'plate2', got None`) after 307 rotation
updates. Reversing the initial jaw polarity and splitting the swing through a 45-degree
intermediate axis produced the same identity-loss boundary. This is not a configuration-load
or motion-goal-resolution failure. The task validates configuration, measured grasp binding,
semantic lift, partial-orientation resolution, and fail-fast held-object identity; successful
vertical placement still requires a plate-grasp/operator dynamics contract that retains the
plate through the swing. Relaxing final placement tolerance would not repair that loss of
identity and would not be valid evidence.

To smoke-test the host-plus-robot composition path independently of task execution:

```bash
python - <<'PY'
from pathlib import Path

from auto_atom.scene_composition import (
    MjcfLayerConfig,
    SceneConfig,
    load_composed_scene,
)

root = Path(".").resolve()
model = load_composed_scene(
    SceneConfig(
        base=root / "assets/xmls/scenes/dishwasher_plate/demo.xml",
        layers=(
            MjcfLayerConfig(path=root / "assets/xmls/robots/robotiq.xml"),
        ),
    )
)
print("loaded ok", "nq=", model.nq, "nu=", model.nu, "ngeom=", model.ngeom)
PY
```

Loading the host alone is useful for asset iteration, but composition acceptance must continue
to use `load_composed_scene` so robot-layer name conflicts and path normalization are exercised.
