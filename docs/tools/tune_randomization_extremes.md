# Tune Randomization Extremes

Interactive tkinter inspector for verifying that configured randomization ranges keep objects and operators within a reasonable workspace. Opens the MuJoCo viewer alongside a control panel that lets you cycle through extreme randomization cases.

**Script:** [examples/tune_randomization_extremes.py](../examples/tune_randomization_extremes.py)

## Usage

```bash
python examples/tune_randomization_extremes.py
python examples/tune_randomization_extremes.py --config-name cup_on_coaster
python examples/tune_randomization_extremes.py --config-name arrange_flowers
```

The default config is `pick_and_place`. The script extracts the `task.randomization` section from the YAML config and builds a set of extreme cases.

## Control panel

The tkinter panel provides:

- **Randomization summary** -- shows all randomized targets with their axis ranges
- **Extreme case selector** -- dropdown to pick a case, with Prev/Next buttons
- **Apply / Reset Default** -- apply the selected case or return to the nominal pose
- **Random Sample** -- draw a fresh random sample uniformly from each configured range
- **Reload Randomization** -- re-read the YAML config from disk (useful when editing ranges in another window)
- **Full Reload** -- rebuild the entire scene and backend from the current config
- **Current Poses** -- live display of each target's position, quaternion, and RPY

## Generated extreme cases

The inspector automatically generates these cases from the config:

| Case | Description |
|---|---|
| `default` | No randomization offset; all targets at nominal pose |
| `all-min` | Every randomized axis at its minimum value simultaneously |
| `all-max` | Every randomized axis at its maximum value simultaneously |
| `<target> <axis>=min` | Single target, single axis at minimum; everything else at default |
| `<target> <axis>=max` | Single target, single axis at maximum; everything else at default |

This covers all corners of the randomization space. If any case pushes an object outside the workspace, off the table, or into collision, the range should be tightened.

## Workflow

1. Configure `task.randomization` in your YAML config (see [Pose Randomization](../task-configuration/randomization.md))
2. Run this inspector to visually verify the extremes
3. Use `all-min` and `all-max` to check the worst-case simultaneous offsets
4. Step through per-axis cases to identify which specific axis causes problems
5. Use `Random Sample` to spot-check typical randomized states
6. Edit your YAML ranges and press `Reload Randomization` to iterate without restarting
