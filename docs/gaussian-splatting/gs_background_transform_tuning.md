# GS Background Offset Tuning

This guide explains how to estimate and interactively tune per-background GS offsets for files under `assets/gs/backgrounds/`.

Each background is one option of the `render_assets/background` config group,
[`aao_configs/render_assets/background/<name>.yaml`](../../aao_configs/render_assets/background/),
which holds its PLY path and offset:

```yaml
# @package env.gaussian_render
# aao_configs/render_assets/background/discover-lab2.yaml
background_ply: ${assets_dir}/gs/backgrounds/discover-lab2.ply
background_transforms:
  discover-lab2: [0.2839, 0.0888, 0.0745]
```

`render=gs` loads `background_1` by default; select another background with
`render_assets/background=<name>`, e.g.
`aao-demo task=press_three_buttons render=gs render_assets/background=discover-lab2`.
To add a background, copy one of these files under the PLY's file stem.

The goal is to shift each background PLY so that:

- the background tabletop height aligns with the MuJoCo tabletop height
- foreground object placement looks physically plausible
- GS masks and overlays reveal less background/foreground occlusion mismatch

## Workflow

Recommended workflow:

1. Run the automatic estimator to get an initial offset.
2. Paste the result into `background_transforms` of that background's `render_assets/background/<name>.yaml`.
3. Launch the interactive tuner and refine the offset while watching RGB, mask, and overlay across one or more cameras.

## 1. Automatic Offset Recommendation

Use [`scripts/render/gs/recommend_gs_background_transforms.py`](../../scripts/render/gs/recommend_gs_background_transforms.py) to estimate an initial xyz offset from the background PLY geometry.

### What the script does

It uses a simple heuristic:

1. Build a histogram over all point `z` values in the background PLY.
2. Find the densest `z` bins.
3. Choose the dense bin nearest a target tabletop height.
4. Collect a narrow `z` band around that peak as a candidate tabletop plane.
5. Estimate the plane center from the median point in that band.
6. Compute the xyz offset needed to align that estimated center with a target tabletop reference pose.

By default the target tabletop reference is:

- tabletop center `xy ~= [0.45, 0.06]`
- tabletop top-surface `z ~= 0.0742`

These defaults match the common GS tabletop convention used by several `demo_gs.xml` scenes in this repository.

### Basic usage

Estimate offsets for every background in `assets/gs/backgrounds/`:

```bash
/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
    scripts/render/gs/recommend_gs_background_transforms.py
```

Estimate a single background:

```bash
/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
    scripts/render/gs/recommend_gs_background_transforms.py \
    assets/gs/backgrounds/discover-lab2.ply
```

Print only a YAML-ready block:

```bash
/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
    scripts/render/gs/recommend_gs_background_transforms.py \
    --yaml
```

Example output:

```yaml
background_transforms:
  background_0: [1.3098, -0.0081, -0.0023]
  discover-lab2: [0.2839, 0.0888, 0.0745]
  simple_room: [-0.0248, 0.1106, 0.0736]
```

### Optional arguments

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `path` | `assets/gs/backgrounds` | One `.ply` file or a directory of `.ply` files |
| `--target-center X Y` | `0.45 0.06` | Target tabletop center in MuJoCo world |
| `--target-z` | `0.0742` | Target tabletop top-surface z |
| `--histogram-bins` | `400` | Number of z histogram bins |
| `--top-dense-bins` | `30` | Number of dense z bins to consider |
| `--min-band-half-width` | `0.02` | Minimum half-width of the candidate tabletop z band |
| `--local-radius-xy` | `1.5` | XY radius used to reject far-away clutter/walls |
| `--min-local-points` | `5000` | Minimum local points required before using the local subset |
| `--yaml` | `false` | Print only YAML-ready output |

Example with a different target reference:

```bash
/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
    scripts/render/gs/recommend_gs_background_transforms.py \
    --target-center 0.45 0.06 \
    --target-z 0.0742 \
    --yaml
```

## 2. Interactive Offset Tuning

Use [`scripts/render/gs/tune_gs_background_transform.py`](../../scripts/render/gs/tune_gs_background_transform.py) to interactively refine a background offset with live preview.

The script updates the GS background offset at runtime and refreshes the rendered preview immediately.

### What the preview shows

For each selected camera it displays:

- `GS color`
- `GS mask`
- `Overlay`

The overlay makes it easier to spot subtle background/foreground occlusion mismatches that may be hard to notice from RGB alone.

### Basic usage

Launch the tuner on one camera:

```bash
/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
    scripts/render/gs/tune_gs_background_transform.py \
    --task press_three_buttons
```

Select a specific background:

```bash
/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
    scripts/render/gs/tune_gs_background_transform.py \
    --task press_three_buttons \
    -- render_assets/background=discover-lab2
```

Tune with a specific camera and smaller step:

```bash
/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
    scripts/render/gs/tune_gs_background_transform.py \
    --task wipe_the_table \
    --camera env1_cam \
    --step 0.002
```

### Multiple cameras

You can preview multiple cameras at once:

```bash
/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
    scripts/render/gs/tune_gs_background_transform.py \
    --task press_three_buttons \
    --camera env1_cam \
    --camera wrist_cam
```

Comma-separated values are also supported:

```bash
... --camera env1_cam,wrist_cam
```

Or preview all configured cameras:

```bash
... --all-cameras
```

### Controls

| Key | Action |
| --- | ------ |
| `a` / `d` | decrease / increase x offset |
| `s` / `w` | decrease / increase y offset |
| `q` / `e` | decrease / increase z offset |
| `[` / `]` | halve / double the step size |
| `r` | reset to the initial offset |
| `p` | print the current offset and a YAML-ready line |
| `x` or `Esc` | exit |

When you press `p`, the script prints a line like:

```yaml
YAML: discover-lab2: [0.2839, 0.0888, 0.0745]
```

You can paste that line back into `background_transforms` in the background's [`aao_configs/render_assets/background/<name>.yaml`](../../aao_configs/render_assets/background/).

### Optional arguments

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `--task` | `press_three_buttons` | Task to compose; always composed with `render=gs` |
| `--camera` | first configured camera | Camera(s) to preview; repeatable |
| `--all-cameras` | `false` | Show all configured cameras |
| `--step` | `0.005` | Initial xyz tuning step in metres |
| `--window-scale` | `0.75` | OpenCV window scale factor |
| `--viewer` | `false` | Also keep the MuJoCo passive viewer open |
| `-- ...` | none | Additional Hydra overrides, e.g. `embodiment=p7_g2p render_assets/background=discover-lab2` |

## Notes

- The automatic estimator is only a starting point. Final alignment should be done visually.
- The most important signal is usually whether the foreground object sits naturally on the background tabletop and whether the GS mask/overlay shows stable visible-region occlusion.
- If a background still looks poor after tuning, the issue may come from the background PLY quality itself rather than offset alone.
