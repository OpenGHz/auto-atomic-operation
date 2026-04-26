# Tune GS Body Mirror

Interactive Qt + OpenCV tool for tuning a `body_mirrors` entry on a Gaussian-Splatting environment with a live side-by-side native MuJoCo + GS render. Every keystroke mutates the in-memory `BodyMirrorSpec`, regenerates the mirrored PLY through `gs_cfg.resolved_body_gaussians()` (cached under `.cache/gs_body_mirrors/`), rebuilds the env's foreground GS renderer, and re-renders both panels through a shared orbit camera.

**Script:** [examples/tune_gs_body_mirror.py](../examples/tune_gs_body_mirror.py)

## When to use

When the physics body and its mirrored gaussians look misaligned (rotation off, mirror plane wrong, post-reflection translation not zeroed), tune by eye instead of editing YAML and re-rendering. Common cases: door panel mirror, articulated handles, asymmetric props.

See [GS Body Transforms & Mirrors](../gaussian-splatting/gs_body_transforms_mirrors.md) for the underlying `body_mirrors` schema.

## Usage

```bash
python examples/tune_gs_body_mirror.py --config-name open_door_airbot_play_back_gs
python examples/tune_gs_body_mirror.py \
    --config-name open_door_airbot_play_back_gs \
    --camera env0_cam --bg-index 3 --width 720 --height 540
```

| Flag             | Default                                  | Purpose                                                        |
|------------------|------------------------------------------|----------------------------------------------------------------|
| `--config-name`  | `open_door_airbot_play_back_gs`          | Hydra task config (must define `env.gaussian_render.body_mirrors`) |
| `--camera`       | first static camera                      | Named MuJoCo camera that seeds the initial orbit pose          |
| `--bg-index`     | `0`                                      | Pin a single background PLY when `background_ply` is a glob/list |
| `--width` / `--height` | `640` / `480`                      | Per-panel resolution                                           |
| `--pos-step`     | `0.01` m                                 | Initial step for position / center nudges                      |
| `--rot-step-deg` | `1.0` deg                                | Initial step for orientation nudges                            |

Hydra overrides can be appended after `--`:

```bash
python examples/tune_gs_body_mirror.py --config-name open_door_airbot_play_back_gs \
    -- env.gaussian_render.minibatch=256
```

## Mouse controls

| Action          | Effect                                  |
|-----------------|-----------------------------------------|
| Left drag       | Orbit (azimuth / elevation)             |
| Right drag      | Zoom (drag down = zoom out)             |
| Middle drag     | Pan                                     |
| Wheel           | Zoom                                    |

## Keyboard controls

| Keys                          | Effect                                                                |
|-------------------------------|-----------------------------------------------------------------------|
| `i` / `k` `j` / `l` `o` / `u` | Post-reflection translate x / y / z ±                                 |
| `y` / `h` `t` / `g` `n` / `m` | Post-reflection rotate roll / pitch / yaw ±                           |
| `Shift` + `i/k j/l o/u`       | Mirror center x / y / z ± (pivot in GS-local coords)                  |
| `1` / `2`                     | Halve / double position step                                          |
| `3` / `4`                     | Halve / double rotation step                                          |
| `Tab`                         | Cycle the selected body                                               |
| `a`                           | Toggle link-all (apply edits to every body in `body_mirrors`)         |
| `f`                           | Cycle mirror axis: GS-local X / Y / Z (clears `body_quat`)            |
| `Shift+M`                     | Toggle mirror on / off for the selected body                          |
| `0`                           | Clear post-reflection orientation to identity                         |
| `p`                           | Print YAML snippet for the current state                              |
| `r`                           | Reset to YAML values                                                  |
| `s`                           | Save snapshot PNG to `/tmp/tuner_<timestamp>.png`                     |
| `b`                           | Print help                                                            |
| `q` / `Esc`                   | Quit                                                                  |

## Workflow

1. Open the tuner on the GS task config.
2. Orbit to a viewpoint where the misalignment is obvious (door edge, handle, prop seam).
3. Press `f` to pick the mirror axis (GS-local X / Y / Z).
4. Use `Shift + i/k j/l o/u` to slide the mirror center until reflected gaussians sit on the physics body.
5. Use `i/k j/l o/u` and `y/h t/g n/m` to fine-tune the post-reflection translate/rotate.
6. Press `p` to print the YAML snippet — paste it into the task config under `env.gaussian_render.body_mirrors`.
7. Press `r` to revert if a tweak made things worse.

## Notes

- The first keystroke of each new mirror config triggers a `resolved_body_gaussians()` regeneration; subsequent identical states hit the on-disk cache instantly.
- The render uses `batch_size=1` and disables depth / mask channels for speed.
- `gs_frame_tuner.py` (background transforms) shares the same key conventions; see [GS Background Transform Tuning](../gaussian-splatting/gs_background_transform_tuning.md).
