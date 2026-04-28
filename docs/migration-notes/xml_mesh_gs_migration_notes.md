# XML / Mesh / GS Migration Notes

This note captures reusable rules for migrating robot XMLs, meshes, Gaussian assets, and scene files so future changes can be done with less trial and error.

Any concrete paths shown below are examples of a normalized asset layout, not mandatory names.

## Scope Guardrails

- Treat the original asset location as a source directory, not the default place to keep editing.
- After assets are migrated into the normalized asset layout, prefer updating the normalized copies rather than repeatedly modifying the original source copies.
- Do not keep bouncing changes between the source directory and the normalized asset directory unless the task explicitly asks for synchronized maintenance of both.
- If a change can be completed entirely inside the normalized asset layout, keep it there.

## Canonical Separation

Keep these asset types conceptually separate:

- robot XML definitions
- scene XML definitions
- robot meshes
- robot Gaussian assets
- background / environment Gaussian assets

A good default structure is:

- `assets/xmls/robots`
- `assets/xmls/scenes/<scene_name>`
- `assets/meshes/<robot_or_object_name>`
- `assets/gs/robots/<robot_name>`
- `assets/gs/backgrounds/<background_name>`

The exact directory names may vary by project, but the separation should remain clear.

## Moving A Robot XML

When moving a robot XML into a canonical robot XML folder:

- update all scene `<include file="...">` references
- re-check `meshdir`
- re-check any secondary `<include>` or `<asset><model file="...">` references
- validate the XML through the final top-level scene, not just by opening the robot XML in isolation
- treat the pre-migration location as the original source location; after migration, the canonical copy should be the one used by scenes

Example:

- before: scene includes a robot XML from a robot-specific source subfolder
- after: scene includes the same robot XML from `assets/xmls/robots/...`

## Relative Paths Over Absolute Paths

Avoid absolute paths in MuJoCo XML unless there is no reasonable alternative.

Prefer the same pattern used by existing portable robot XMLs:

- use relative `meshdir`
- keep auxiliary included asset XMLs near the robot XML
- validate from the final scene entrypoint

Reason:

- absolute paths break portability across machines
- MuJoCo path resolution often depends on the top-level scene load context
- a path that looks correct relative to the included XML can still fail at runtime if it is inconsistent with how the scene is loaded

## `meshdir` Rule

When a robot XML is included by a scene XML:

- choose `meshdir` so the final scene can resolve meshes correctly
- do not assume the included XML will always be resolved in isolation
- mimic the relative-path style of already-working portable robot XMLs in the project

Example:

```xml
<compiler angle="radian" meshdir="../../meshes/example_robot" autolimits="true"/>
```

The exact number of `..` segments depends on the project layout.

## Scene Replacement Rule

When replacing one robot with another in a scene:

- prefer creating a parallel scene file first instead of overwriting the original
- keep the original scene as a reference

Example:

- keep `demo.xml`
- add `demo_robot_b.xml`

This makes it easier to compare behavior, keyframes, and path resolution.

## Fixed-Base Robot vs Free-Base Robot

This is one of the most important migration distinctions.

Free-base robot:

- can often be dropped into a scene and repositioned via freejoint state
- is often easier to reuse in table-centered demos

Fixed-base robot:

- cannot be repositioned the same way
- usually requires moving the robot root body or wrapping it in an offset body
- should not force task objects or the table to move if the request is to preserve scene layout

## Preferred Repositioning Strategy

If the scene layout must stay unchanged:

- move the robot root
- do not move the table, target objects, or cameras unless requested

Typical approaches:

1. Add a `pos="x y z"` offset on the robot root body.
2. Wrap the robot in a parent body with the desired transform.

Prefer the smallest structural change that preserves scene semantics.

## Keyframe Migration Rule

When swapping the included robot model:

- `qpos` layout may change
- `ctrl` layout may change
- actuator count may change
- named joints and sites may change

So every scene keyframe must be re-checked after robot replacement.

Do not reuse keyframes blindly across different robot definitions.

## Validation Checklist

Before finishing a migration:

1. Load the final top-level scene with MuJoCo.
2. Confirm all included XML files resolve.
3. Confirm all mesh files resolve.
4. Confirm the robot appears in the intended location relative to the unchanged scene.
5. Confirm `qpos` and `ctrl` dimensions match the new robot model.
6. If Gaussian assets were moved, confirm robot GS and background GS still follow the intended split.

Recommended validation pattern:

```bash
<python> - <<'PY'
import mujoco
m = mujoco.MjModel.from_xml_path('path/to/scene.xml')
print('loaded ok', 'nq=', m.nq, 'nu=', m.nu)
PY
```

Then optionally open the viewer:

```bash
cd path/to/scene_dir
python -m mujoco.viewer --mjcf scene.xml
```

## Common Failure Modes

- Continuing to edit the original source asset directory after a canonical migrated copy already exists
- Using absolute `meshdir` paths
- Forgetting to update scene `<include>` paths after moving a robot XML
- Keeping old keyframes after swapping the robot model
- Moving the task table or objects when only the robot should be repositioned
- Mixing robot Gaussian assets with background/environment Gaussian assets
- Verifying the robot XML alone, but not the final top-level scene

## Gaussian Asset Rule

Keep robot GS and scene/background GS separate.

Good pattern:

- robot GS under `assets/gs/robots/...`
- environment GS under `assets/gs/backgrounds/...`

Specific names are repo-dependent; the separation principle is the important part.

## Suggested Future Workflow

1. Identify whether the task can stay within the normalized asset layout or whether it truly requires changes in the original source location.
2. Move or normalize the robot XML first.
3. Fix scene includes.
4. Fix `meshdir` and any auxiliary asset references.
5. If robot type changed, re-check keyframes and controls.
6. If layout must remain unchanged, move the robot root rather than scene objects.
7. Validate by loading the final scene.
8. Only then update GS references or viewer examples if needed.

## Example-Only References

Examples from the current project may include:

- moving a robot XML into `assets/xmls/robots`
- keeping robot meshes under `assets/meshes/<robot_name>`
- keeping robot GS under `assets/gs/robots/<robot_name>`
- creating a sibling scene such as `demo_robot_variant.xml`

Treat those as examples, not mandatory naming rules.
