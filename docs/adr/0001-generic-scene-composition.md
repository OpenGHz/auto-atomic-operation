---
status: accepted
---

# Generic scene composition contract

Scene construction uses a pure-data `SceneConfig` containing one host MJCF
document and an ordered list of typed layers.  Pre-authored MJCF and packaged
asset assemblies are compiled through adapters into self-contained
contributions; the host composer alone performs namespace/collision validation,
merge, digesting, and temporary-file lifecycle.  `EnvConfig`, the MuJoCo basis,
and the viewer depend only on this generic contract, never on a vendor asset
family.

We rejected a UniDoor-specific field and loader branch, and treating robots as
a separate configuration seam.  Both alternatives make the interface shallow
and force every later asset family to duplicate merge and lifecycle logic.
This is intentionally a breaking change: `EnvConfig.scene` is the only scene
configuration surface. Asset package descriptors use relative artifact
references and explicit mechanism axes/frames; legacy combination XML remains
a regression oracle, not a runtime dependency.

The first renderer is MJCF, but the contribution and semantic-export contract
is intentionally backend-neutral so a future renderer can consume the same
package/recipe without changing task configuration.

Mechanism axis, pivot, limits and dynamics belong to the versioned assembly
template. The UniDoor runtime adapter rejects packages that omit explicit
joint axes or dynamics; only the offline migration tool may infer values from
source handedness, and it records that provenance in the generated package.
The generic composer never guesses physics.

For MJCF singleton sections (`option`, `compiler`, `visual`, and similar), the
host scene is authoritative and a layer only fills attributes the host omitted.
This is an intentional breaking clarification over the former duplicate-tag
inlining behavior: task-level simulation settings cannot be silently replaced
by whichever robot XML happens to be listed last.
