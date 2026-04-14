# Tactile Prefix Binding

## Overview

Tactile sensors are attached to gripper fingers via MuJoCo's `<attach>` mechanism with a `prefix` attribute. This prefix propagates to all site/sensor names inside the attached model, and is used by the framework to:

1. **Group sensor data** into per-finger panels (`TactileSensorManager`)
2. **Bind panels to operators** via `tactile_prefixes` in `OperatorBinding`

## How Prefixes Flow

```
XML model                          Runtime
─────────                          ───────
<attach prefix="left_"/>    →    site: left_touch_point_01
                                  sensor: left_force_sensor_01
                                  ↓
                            TactileSensorManager extracts prefix "left_"
                                  ↓
                            _find_operator_for_tactile_panel("left_")
                                  ↓
                            matches OperatorBinding.tactile_prefixes: ["left_", ...]
```

### Step 1: XML Attachment

In the gripper XML (e.g. `xf9600_gripper.xml`):
```xml
<body name="right_tactile" ...>
  <attach model="tactile_sensor" prefix="right_"/>
</body>
<body name="left_tactile" ...>
  <attach model="tactile_sensor" prefix="left_"/>
</body>
```

The shared tactile sensor model (`xf9600_tactile_sensor.xml`) defines sites named `touch_point_01`, `touch_point_02`, etc. After attachment, MuJoCo prefixes them to `right_touch_point_01`, `left_touch_point_01`, etc.

### Step 2: Panel Discovery

`TactileSensorManager._extract_prefix_from_site_name()` scans all sites containing `"touch_point"` and extracts the prefix (everything before `"touch_point"`). Each unique prefix becomes a **panel** — one panel per finger.

### Step 3: Panel-to-Operator Binding

`_find_operator_for_tactile_panel()` in `mujoco_env.py` matches each panel prefix against the `tactile_prefixes` list in `OperatorBinding`:

```python
# For each panel prefix (e.g. "left_"):
for op in self._operators.values():
    if not op.tactile_prefixes:
        if len(self._operators) == 1:
            return op  # single operator, auto-assign all panels
    elif any(panel_prefix.startswith(pfx) for pfx in op.tactile_prefixes):
        return op  # prefix match found
```

**Rules:**
- If `tactile_prefixes` is empty and there's only one operator → all panels auto-bind to it
- If `tactile_prefixes` is specified → only panels whose prefix starts with a listed value are bound
- Multi-operator setups (e.g. dual-arm) **require** explicit `tactile_prefixes` to disambiguate

### Step 4: Observation Key Generation

`_group_tactile_by_component()` in `mujoco_env.py` builds observation keys from the panel prefix and operator's `eef_output_name`:

```python
# panel_prefix = "left_" → panel_label = "left"
# eef_output_name = "eef"
# → key = "eef_left"  (non-structured)
# → key = "tactile/eef/left/points"  (structured)
```

## YAML Configuration

```yaml
env:
  operators:
    arm:
      eef_actuators: [fingers_actuator]
      eef_output_name: eef
      tactile_prefixes: ["left_", "right_"]   # must match XML <attach prefix="..."/>
```

## Renaming Prefixes

When changing tactile attachment prefixes in the XML (e.g. `left_`/`right_` → `finger1_`/`finger2_`), you must update **both**:

1. **XML model** — the `<attach prefix="..."/>` attribute
2. **YAML config** — the `tactile_prefixes` list in the corresponding operator

```yaml
# Before
tactile_prefixes: ["left_", "right_"]

# After (matching new XML prefixes)
tactile_prefixes: ["finger1_", "finger2_"]
```

Observation keys will change accordingly (e.g. `eef_left` → `eef_finger1`), so downstream consumers (data collection, policy evaluation) may also need updating.

## Panel Naming Convention

`TactileSensorManager._parse_panel_meta()` splits the prefix on `_` to derive `arm` and `finger` labels for sorting and visualization:

| Prefix | arm | finger | Visualization title |
|--------|-----|--------|-------------------|
| `left_` | (single token) | `0` | `left / 0` |
| `right_` | (single token) | `0` | `right / 0` |
| `arm_left_` | `arm` | `left` | `arm / left` |
| `finger1_` | (single token) | `0` | `finger1 / 0` |
| `dual_finger1_` | `dual` | `finger1` | `dual / finger1` |

The last `_`-separated token is used as the finger label when there are 2+ tokens; otherwise the finger defaults to `"0"`.
