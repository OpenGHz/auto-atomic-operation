# aao_configs: flat task files → Hydra config groups

`aao_configs/` used to hold about 70 flat YAML files, one per runnable variant
(`<task>[_<robot>][_gs].yaml`) plus building blocks (`basis_*`, `*_mixin`,
`*_vars`, `*_gs`). It is now **one primary config, `aao_configs/config.yaml`,
plus config groups**. A run is selected by group choices instead of a file
name:

```bash
# before
aao-demo --config-name cup_on_coaster_gs_airbot_p7
# after
aao-demo task=cup_on_coaster embodiment=p7_g2p
```

There are **no compatibility aliases**: every `--config-name <name>` invocation
must be rewritten with the [mapping table](#old-config-name--new-overrides)
below.

## Why

- **Per-variant duplication.** Every robot × renderer combination of a task
  was its own file. A GS variant restated the robot's GS block and the scene's
  PLYs, a robot variant restated the scene, and adding one robot or one
  renderer multiplied files across tasks.
- **List fields were not mergeable.** `env.cameras` and `env.scene.layers`
  were lists, which Hydra replaces wholesale. A variant that needed one more
  camera or a different robot layer had to restate the entire list, and every
  camera entry repeated `width: ${cam_width}`, `enable_depth: ${enable_depth}`,
  ... from `common_vars`.
- **Robot properties were scattered.** The robot MJCF layer, home pose, wrist
  camera name, top-down grasp orientation, and IK backend were split between
  `basis_*` files and individual task files.
- **`+backend=warp` needed a `+`.** Each `basis_*` re-asserted its CPU backend
  builder in `_self_`, so a plain `backend=warp` failed with "Could not
  override 'backend'"; the GPU backend and the EGL environment (`+env=gl`) had
  to be appended.

## New layout

`aao_configs/config.yaml` composes these groups in order. Later entries win,
and command-line overrides win over everything:

| Group | Options | Default | Owns |
|---|---|---|---|
| `simulator` | `mujoco`, `mock` | `mujoco` | Environment shell |
| `execution` | `physical`, `object_only` | `physical` | Execution mode block |
| `observation` | `default`, `rgb_only` | `default` | Sensor streams and camera defaults (`observation.camera.*`) |
| `camera_layout` | `all`, `operator_only`, `no_operator` | `all` | Camera roles kept |
| `scene` | `scene/*.yaml` | chosen by the task | `scene_name`, static cameras, scene asset layers |
| `embodiment` | `embodiment/*.yaml` | chosen by the task | Robot layer, operators, IK/backend, home pose, wrist camera |
| `task` | `task/*.yaml` | `pick_and_place` | Stages, objects, randomization, viewer |
| `render` | `mujoco`, `gs` | `mujoco` | Renderer |
| `render_assets/{embodiment,scene,task}` | `<render>/<name>.yaml` | auto, optional | Per-renderer asset bindings (GS PLYs) |
| `adapt` | `<task>/<embodiment>.yaml` | auto, optional | Task × embodiment tuning |
| `randomization` | `<task>/<preset>` | none | Opt-in reset-randomization preset for one task |
| `backend` | `cpu`, `warp` | `cpu` | Physics backend |
| `platform` | `egl` | none | Headless-rendering environment variables |

Conventions:

- A task selects its scene and default embodiment with `override /scene:` and
  `override /embodiment:` in its own `defaults` list; `embodiment=...` on the
  command line still wins.
- `render`, `backend`, and `platform` compose after the task, so they are
  user-level choices a task cannot override.
- `adapt/<task>/<embodiment>.yaml` and
  `render_assets/<axis>/<render>/<name>.yaml` are specialization slots resolved
  from the final group choices and silently skipped when no file exists.
- `_`-prefixed files and directories (`embodiment/_p7.yaml`,
  `task/_press_button.yaml`, `adapt/_open_door/`) are shared fragments
  included through `defaults`; they are never selected directly.
- In the composed (raw) config, `env.cameras` and `env.scene.layers` are dicts
  keyed by slot (`wrist`, `env0`, `robot`, `door`, ...). A later config
  replaces one slot or removes it with `null`.
  `auto_atom.execution_config.prepare_task_config_for_instantiation` flattens
  them into lists in slot order. Every camera inherits the
  `observation.camera` defaults, and cameras whose role is not in
  `camera_layout.roles` are dropped.
- `env.initial_joint_positions.<joint>: null` drops one inherited joint entry
  (for example, the Robotiq freejoint home of `task/open_door.yaml` on a P7).

See [Reusing & Creating Tasks](../task-configuration/reusing_and_creating_tasks.md)
for where new tasks, robots, adaptations, and GS assets go.

## Command-line changes

| Before | After |
|---|---|
| `--config-name <name>` | `task=<task> [embodiment=<embodiment>] [render=gs]` (see the table below) |
| `+backend=warp` | `backend=warp` |
| `+env=gl` | `platform=egl` |
| `bg3dgs_name=<name>` | `render_assets/background=<name>` |
| `cam_width=`, `cam_height=` | `observation.camera.width=`, `observation.camera.height=` |
| `enable_color=`, `enable_depth=`, `enable_mask=`, `enable_heat_map=` | `observation.camera.enable_color=`, ... |
| `+test=test` | `observation=rgb_only` |
| `+test=open_the_door` | unchanged (the file was rewritten for the slot-keyed cameras) |
| `env.cameras=[]` | `~env.cameras` (cameras are a slot dict now) |
| `aao-unidoor-sweep --config-name <name>` | `aao-unidoor-sweep --task <task> --embodiment <embodiment>` |
| argparse example scripts: `--config-name <name>` | `--task <task>` plus Hydra overrides (see each tool page) |

Runtime keys such as `+rounds=`, `+max_updates=`, `+use_input=`, and
`+print_updates=` are unchanged.

## Old config name → new overrides

| Old `--config-name` | New overrides |
|---|---|
| `arrange_flowers` | `task=arrange_flowers` |
| `arrange_flowers_gs` | `task=arrange_flowers render=gs` |
| `arrange_flowers_gs_airbot_p7` | `task=arrange_flowers embodiment=p7_g2p render=gs` |
| `close_drawer` | `task=close_drawer` |
| `close_hinge_door` | `task=close_hinge_door` |
| `cup_on_coaster` | `task=cup_on_coaster` |
| `cup_on_coaster_gs` | `task=cup_on_coaster render=gs` |
| `cup_on_coaster_gs_airbot_p7` | `task=cup_on_coaster embodiment=p7_g2p` |
| `cup_on_coaster_airbot_p7_umi` | `task=cup_on_coaster embodiment=p7_v3_umi_v3` |
| `dishwasher_plate` | `task=dishwasher_plate` |
| `hang_toothbrush_cup` | `task=hang_toothbrush_cup` |
| `hang_toothbrush_cup_gs` | `task=hang_toothbrush_cup render=gs` |
| `microwave_sweet_potato_umi_v3` | `task=microwave_sweet_potato` |
| `mock` | `task=mock` |
| `policy_eval_mock` | `task=policy_eval_mock` |
| `open_door` | `task=open_door` |
| `open_door_airbot_play_g2p` | `task=open_door embodiment=airbot_play_g2p` |
| `open_door_airbot_play_gs` | `task=open_door embodiment=airbot_play_g2p render=gs` |
| `open_door_airbot_play_back_gs` | `task=open_door_back render=gs` |
| `open_door_p7_ik` | `task=open_door embodiment=p7_xf9600` |
| `open_door_p7_v3_umi_v3` | `task=open_door embodiment=p7_v3_umi_v3` |
| `open_door_unidoor_p7_v3_umi_v3` | `task=open_door_unidoor` |
| `open_door_unidoor_p7_v4_umi_v3` | `task=open_door_unidoor embodiment=p7_v4_umi_v3` |
| `open_drawer` | `task=open_drawer` |
| `open_hinge_door` | `task=open_hinge_door` |
| `pick_and_place` | `task=pick_and_place` |
| `pick_and_place_franka` | `task=pick_and_place embodiment=franka_robotiq` |
| `pick_and_place_umi_v3` | `task=pick_and_place embodiment=umi_v3_mocap` |
| `pick_and_place_xf9600` | `task=pick_and_place embodiment=xf9600_mocap` |
| `place_blocks_on_disk_airbot_play_g2` | `task=place_blocks_on_disk` |
| `press_blue_button` | `task=press_blue_button render=gs` |
| `press_green_button` | `task=press_green_button render=gs` |
| `press_pink_button` | `task=press_pink_button render=gs` |
| `press_blue_button_airbot_p7` | `task=press_blue_button embodiment=p7_g2p render=gs` |
| `press_green_button_airbot_p7` | `task=press_green_button embodiment=p7_g2p render=gs` |
| `press_pink_button_airbot_p7` | `task=press_pink_button embodiment=p7_g2p render=gs` |
| `press_three_buttons` | `task=press_three_buttons` |
| `press_three_buttons_gs` | `task=press_three_buttons render=gs` |
| `rack_plate_p7_v4_umi_v3` | `task=rack_plate` |
| `stack_color_blocks` | `task=stack_color_blocks` |
| `stack_color_blocks_gs` | `task=stack_color_blocks render=gs` |
| `wipe_the_table` | `task=wipe_the_table` |
| `wipe_the_table_gs` | `task=wipe_the_table render=gs` |
| `wipe_the_table_gs_airbot_p7` | `task=wipe_the_table embodiment=p7_g2p render=gs` |

## Removed building blocks

| Old file(s) | Replacement |
|---|---|
| `basis.yaml` | `simulator/mujoco.yaml` |
| `basis_mocap_eef` | `embodiment/robotiq_mocap.yaml` (shared part: `embodiment/_mocap_gripper.yaml`) |
| `basis_mocap_eef_xf9600` | `embodiment/xf9600_mocap.yaml` |
| `basis_mocap_eef_umi_v3` | `embodiment/umi_v3_mocap.yaml` |
| `basis_franka` | `embodiment/franka_robotiq.yaml` |
| `basis_p7_xf9600` | `embodiment/p7_xf9600.yaml` (shared P7 part: `embodiment/_p7.yaml`) |
| `basis_p7_xf9600_composable` | `embodiment/p7_xf9600_composable.yaml` |
| `basis_p7_g2p` | `embodiment/p7_g2p.yaml` |
| `basis_p7_v3_umi_v3` | `embodiment/p7_v3_umi_v3.yaml` (new sibling: `embodiment/p7_v4_umi_v3.yaml`) |
| `basis_airbot_play_g2p` | `embodiment/airbot_play_g2p.yaml` (shared part: `embodiment/_airbot_play.yaml`) |
| `basis_airbot_play_g2` | `embodiment/airbot_play_g2.yaml` |
| `basis_airbot_play_xf9600` | `embodiment/airbot_play_xf9600.yaml` |
| `basis_xf9600` (fixed-arm base) | `embodiment/_fixed_arm.yaml` (fragment) |
| `common_vars`, `compatible_vars` | `observation/default.yaml` (`observation.camera.*`); the compatibility variables are gone |
| `gs_mixin` | `render/gs.yaml` + `render_assets/background/<name>.yaml` |
| `robotiq_gs`, `airbot_play_gs`, `airbot_g2p_gs`, `airbot_xf9600_gs` | `render_assets/embodiment/gs/<embodiment>.yaml` |
| `*_gs_mixin` (e.g. `press_three_buttons_gs_mixin`) and the PLY blocks of `<task>_gs` files | `render_assets/scene/gs/<scene>.yaml` |
| `press_button_basis` | `task/_press_button.yaml` |
| `open_door_airbot_play_back_mixin` | `task/open_door_back.yaml` + `adapt/open_door_back/airbot_play_g2p.yaml` (+ `render_assets/task/gs/open_door_back.yaml`) |
| Robot-specific waypoints inside `<task>_<robot>` files | `adapt/<task>/<embodiment>.yaml` (shared: `adapt/_<task>/<family>.yaml`) |
| `env/gl.yaml` | `platform/egl.yaml` |
| `test/test.yaml` | `observation/rgb_only.yaml` |

## Python API changes

- `auto_atom.config_loader.compose_task_config(task=None, overrides=None,
  config_dir=None, *, return_hydra_config=False)` composes `config.yaml` with
  `task=<task>` prepended. Use it instead of `initialize_config_dir(...)` +
  `compose(config_name=<old name>)`.
- `PRIMARY_CONFIG_NAME == "config"`; `task_overrides(task, overrides)` returns
  `["task=<task>", *overrides]`.
- `load_task_file_hydra(task, config_dir=None, overrides=None)`: the first
  parameter is now a task name.
- `RemotePolicyEvaluator.from_config(task, overrides, ...)`: the first argument
  is a task name; embodiment / render go into `overrides`.
- Every `@hydra.main` entry point uses `config_name="config"`, so
  `HydraConfig.get().job.config_name` is always `"config"`. Name per-run
  outputs with `auto_atom.runner.common.get_run_name()` (inside
  `@hydra.main`) or `describe_run(choices)` (from
  `HydraConfig.get().runtime.choices`, or `cfg.hydra.runtime.choices` when
  composed with `return_hydra_config=True`).
- Code that reads `cfg.env.cameras` or `cfg.env.scene.layers` as lists must
  read them from `prepare_task_config_for_instantiation(cfg)` (or call
  `normalize_env_collections`) first.

## Behavioral notes

Composed configs are equivalent to the old ones, except:

- The dead top-level `operators:` list was removed from `open_door`; it was
  ignored by `TaskFileConfig`.
- Mock tasks (`task=mock`, `task=policy_eval_mock`) now carry the default
  `execution` block, which is identical to the schema defaults.
- GS `env.gaussian_render.background_transforms` contains only the selected
  background's entry instead of the table of all backgrounds. The open_door GS
  runs keep an unused `background_1` entry next to their `wall` / `inside`
  transforms.
- The camera order of `cup_on_coaster` on `p7_g2p` and `p7_v3_umi_v3` is now
  wrist, `env1_cam`, `env0_cam` (it was wrist, `env0_cam`, `env1_cam`).
  Consumers that index cameras by position must be checked.
- `press_*_button` tasks default to native MuJoCo rendering. Pass
  `render=gs` for the old default.

Output naming: recordings, comparison images, benchmark results, and run
summaries are named by the **run name** `task__embodiment[__render]` (the
render part is omitted for native MuJoCo) instead of the config name, for
example `pick_and_place__robotiq_mocap` or `press_blue_button__p7_g2p__gs`.
`summary.json` records it as `run_config.run_name` together with the selected
group `choices`. Existing artifacts named after old config names (for example
`outputs/records/demos/<config_name>.npz`) are not renamed; pass their name
explicitly (e.g. `+replay.demo_name=<old name>`) or rename them.

## Verification

Check that a task and its variants compose and resolve to the expected
embodiment and workflow:

```bash
aao-info --no-progress                    # every task x embodiment variant + its aao-demo command
aao-info <task> --verbose                 # one task; reports skipped variants and composition errors
aao-demo task=<task> [embodiment=<e>] [render=gs] --info defaults   # which group / adapt / render_assets files composed
aao-demo task=<task> [embodiment=<e>] [render=gs] --cfg job         # the composed config
```

Parity with the flat layout was checked before the old files were removed:

- **Static check.** All 44 old configs were composed and prepared with Hydra
  and compared to their new `task=… [embodiment=…] [render=gs]` equivalents.
  They are equal except for the intended differences listed under
  [Behavioral notes](#behavioral-notes).
- **Runtime check.** Old (main worktree) and new configs were run side by side
  with `aao-demo … env.viewer=null env.batch_size=1 ++max_updates=300` (or
  `600`). `final_success`, `completed_stages`, and `completion_steps` are
  identical for 22 variants: `mock`; `pick_and_place` on `robotiq_mocap`,
  `xf9600_mocap`, and `umi_v3_mocap`; `open_door` on `robotiq_mocap`,
  `p7_xf9600`, `p7_v3_umi_v3`, and `airbot_play_g2p`; `cup_on_coaster` on
  `robotiq_mocap`, `p7_g2p`, and `p7_v3_umi_v3`; `dishwasher_plate`,
  `microwave_sweet_potato`, `press_three_buttons`, `rack_plate`,
  `place_blocks_on_disk`, `arrange_flowers`, `close_drawer`,
  `close_hinge_door`, `open_drawer`, `open_hinge_door`, `stack_color_blocks`,
  and `wipe_the_table`.
- **GS runtime check.** In the pixi `gs` environment, 13 GS variants were run
  side by side with the same `aao-demo` settings and produced identical
  results: `press_blue_button` (`robotiq_mocap`, `p7_g2p`),
  `press_pink_button` (`p7_g2p`), `press_three_buttons`, `cup_on_coaster`,
  `arrange_flowers` (`robotiq_mocap`, `p7_g2p`), `stack_color_blocks`,
  `wipe_the_table` (`robotiq_mocap`, `p7_g2p`), `open_door` (`airbot_play_g2p`),
  `open_door_back`, and `hang_toothbrush_cup`.
- **GS image check.** One post-reset observation was captured from 12 GS
  variants with the old and new layouts; every color and heat-map image is
  pixel-identical. The two open_door variants pick their wall background and
  inside offset randomly on every construction, so they were compared with
  `wall_name=wall0 ~env.gaussian_render.background_transform_randomization`.
- **Not run.** Franka (`mink` was not installed) and UniDoor (the asset payload
  was not present) could not be run in the verification environment; each
  failed identically on the old and new layouts. Native-rendered
  `hang_toothbrush_cup` crashed on both layouts at construction time (fixed separately: MuJoCo
  ran out of constraint memory for the contacts of the operator buried at
  `qpos0`); after the fix it runs, but its `pick_cup` stage does not succeed (the
  GS variant fails the same way on both layouts). The GS variant was later fixed
  separately by stiffening the Robotiq mocap weld; the native scene layout still
  does not complete `pick_cup`.
