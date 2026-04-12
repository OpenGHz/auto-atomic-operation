# Panel XML Assembly

[`examples/generate_panel_assembly_xml.py`](../examples/generate_panel_assembly_xml.py) can generate a final MJCF scene by taking:

- one panel/base XML
- one 2-D layout definition
- one object XML per slot

and mounting all configured objects onto the panel as fixed child bodies.

This is useful when you have a mechanical layout drawing such as a `7 x 3` panel grid and want to describe the arrangement in YAML instead of hand-editing a large MuJoCo XML.

## Related files

- Script: [`examples/generate_panel_assembly_xml.py`](../examples/generate_panel_assembly_xml.py)
- Core generator: [`auto_atom/utils/panel_xml_builder.py`](../auto_atom/utils/panel_xml_builder.py)
- Example config: [`examples/panel_assembly/layout_7x3.yaml`](../examples/panel_assembly/layout_7x3.yaml)
- Example panel XML: [`examples/panel_assembly/demo_panel_base.xml`](../examples/panel_assembly/demo_panel_base.xml)
- Example object XMLs:
  - [`examples/panel_assembly/objects/red_switch.xml`](../examples/panel_assembly/objects/red_switch.xml)
  - [`examples/panel_assembly/objects/green_knob.xml`](../examples/panel_assembly/objects/green_knob.xml)
  - [`examples/panel_assembly/objects/amber_lamp.xml`](../examples/panel_assembly/objects/amber_lamp.xml)

## Basic usage

Generate the final panel XML from a YAML config:

```bash
python examples/generate_panel_assembly_xml.py \
  examples/panel_assembly/layout_7x3.yaml
```

Override the output path if needed:

```bash
python examples/generate_panel_assembly_xml.py \
  examples/panel_assembly/layout_7x3.yaml \
  --output outputs/panel_assembly_demo.xml
```

Preview the generated XML in MuJoCo:

```bash
python -m mujoco.viewer --mjcf=outputs/panel_assembly_demo.xml
```

## YAML structure

The config has four main parts:

```yaml
panel:
  xml: demo_panel_base.xml
  attach_to_body: panel_mount

output_xml: generated/demo_panel_with_objects.xml

layout:
  coordinate_scale: 0.001
  face_axes: [x, z]
  x_coords: [-349.80, -233.20, -116.60, 0.0, 116.60, 233.20, 349.80]
  y_coords: [120.0, 0.0, -120.0]
  base_pos: [0.0, 30.0, 0.0]
  default_quat: [1.0, 0.0, 0.0, 0.0]
  remove_root_joints: true

placements:
  - - xml: objects/red_switch.xml
    - xml: objects/green_knob.xml
    - xml: objects/amber_lamp.xml
    - null
    - xml: objects/red_switch.xml
    - xml: objects/green_knob.xml
    - xml: objects/amber_lamp.xml
  - - ...
  - - ...
```

## Field reference

### `panel`

| Field | Required | Meaning |
|---|---|---|
| `xml` | yes | Path to the base panel MJCF |
| `attach_to_body` | no | Name of the body under `worldbody` that receives all slot bodies. If omitted, slots are attached directly under `worldbody` |

### `output_xml`

Final generated XML path. If it is relative, it is resolved relative to the YAML file location.

### `layout`

| Field | Required | Meaning |
|---|---|---|
| `coordinate_scale` | no | Scale factor applied to `x_coords`, `y_coords`, `base_pos`, and `pos_offset`. Useful when drawings are in `mm` and XML is in `m` |
| `face_axes` | no | Which local axes represent the 2-D panel face. Default is `[x, z]` |
| `x_coords` | yes | Column coordinates |
| `y_coords` | yes | Row coordinates |
| `base_pos` | no | Base offset of the whole grid before row/column offsets are applied |
| `default_quat` | no | Default slot orientation quaternion `(w x y z)` |
| `remove_root_joints` | no | Whether to remove top-level `joint` / `freejoint` from mounted object bodies so they become fixed to the panel |

### `placements`

`placements` is a 2-D matrix.

- Number of rows must equal `len(layout.y_coords)`
- Number of columns in each row must equal `len(layout.x_coords)`
- `null` means that slot is empty
- A string is treated as a shorthand for `xml: <that string>`
- A mapping allows per-slot overrides

Example:

```yaml
placements:
  - - xml: objects/red_switch.xml
    - xml: objects/green_knob.xml
      pos_offset: [0.0, 0.0, 0.005]
    - null
```

Supported per-slot fields:

| Field | Required | Meaning |
|---|---|---|
| `xml` | yes | Object XML path |
| `body_name` | no | Which top-level body in the object XML should be mounted |
| `slot_name` | no | Name of the generated slot body |
| `mounted_body_name` | no | Name of the mounted object root body |
| `pos_offset` | no | Extra local translation on top of the grid position |
| `quat` | no | Slot-local quaternion `(w x y z)` overriding `layout.default_quat` |
| `remove_root_joints` | no | Slot-level override of `layout.remove_root_joints` |

## Coordinate mapping

The generator interprets the layout as a 2-D face and maps it into the panel body's local coordinates.

For example:

```yaml
layout:
  coordinate_scale: 0.001
  face_axes: [x, z]
  x_coords: [-349.80, 0.0, 349.80]
  y_coords: [120.0, 0.0, -120.0]
  base_pos: [0.0, 30.0, 0.0]
```

means:

- column coordinates are applied along local `x`
- row coordinates are applied along local `z`
- all slots are shifted by `0.03` along local `y`
- because `coordinate_scale: 0.001`, the drawing values are treated as millimeters and converted to meters

This is helpful when your drawing uses panel coordinates but the panel body itself is rotated or mounted somewhere else in a larger scene.

## Requirements for object XMLs

Each object XML must provide a mountable body in one of these forms:

1. root element is directly a `<body>`
2. root element is `<mujoco>` and its `<worldbody>` contains exactly one top-level `<body>`
3. root element is `<mujoco>` and you explicitly select the target top-level body with `body_name`

Typical object XML:

```xml
<mujoco model="red_switch">
  <worldbody>
    <body name="red_switch">
      <geom name="red_switch_base" type="box" />
    </body>
  </worldbody>
</mujoco>
```

If the object XML includes a `freejoint` or top-level `joint`, leave `remove_root_joints: true` so the generated object becomes fixed to the panel.

## What the generator does automatically

### 1. Mounts objects as fixed children

For each configured slot, the generator creates:

- one slot body at the computed panel position
- one mounted object body underneath it

This gives a clean hierarchy such as:

```xml
<body name="panel_mount">
  <body name="panel_slot_r0_c0" pos="...">
    <body name="red_switch_r0_c0">
      ...
    </body>
  </body>
</body>
```

### 2. Merges required MJCF sections

It merges object-side dependencies such as:

- `include`
- `default`
- `asset`
- `contact`
- `equality`
- `tendon`
- `actuator`
- `sensor`
- `custom`

This means materials, meshes, defaults, sensors, and similar definitions from object XMLs are brought into the final generated XML.

### 3. Rewrites relative asset paths

If the source panel XML or object XML contains relative `file` paths, they are rewritten relative to the output XML location so meshes and includes still resolve after generation.

### 4. Renames per-instance runtime objects

MuJoCo requires names such as `body`, `geom`, `site`, and `joint` to be unique where applicable. When the same object XML is mounted multiple times, the generator automatically namespaces runtime names per slot.

Example:

- source name: `red_switch_base`
- generated names: `red_switch_base__r0_c0`, `red_switch_base__r0_c3`, ...

This avoids errors like:

```text
XML Error: Error: repeated name 'red_switch_base' in geom
```

Asset and default names are intentionally not namespaced, so shared materials and classes can be reused safely across repeated instances.

## Example: 7 x 3 panel

The example config in [`examples/panel_assembly/layout_7x3.yaml`](../examples/panel_assembly/layout_7x3.yaml) encodes the layout:

- columns: `-349.80`, `-233.20`, `-116.60`, `0`, `116.60`, `233.20`, `349.80`
- rows: `120`, `0`, `-120`
- scale: `0.001`

So the final slot positions become:

- left to right: `-0.3498` to `0.3498`
- top to bottom: `0.12`, `0.0`, `-0.12`

## Troubleshooting

### MuJoCo reports repeated names

If you are using the current generator, repeated `geom/site/body/joint` names from duplicated object XMLs should already be handled automatically.

If a repeated-name error still appears, check whether:

- the duplicated name comes from `asset` or `default` sections that are genuinely conflicting
- two different object XMLs define different content under the same asset/default name

In that case, rename the conflicting material/default/mesh in the source XMLs.

### Object is not fixed to the panel

Check:

- `layout.remove_root_joints: true`
- the selected object root body is the one containing the top-level `freejoint`

### Slot count does not match the drawing

Check:

- number of rows in `placements`
- number of columns in each row
- lengths of `x_coords` and `y_coords`

### Layout direction is wrong

Adjust:

- `layout.face_axes`
- `layout.base_pos`
- slot-level `quat`

If your drawing is defined in panel-face coordinates but the panel body uses another local axis convention, `face_axes` is usually the first thing to fix.
