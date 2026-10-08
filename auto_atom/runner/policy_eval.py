"""Console entry point for policy evaluation."""

from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from auto_atom.policy_eval import (
    ConfigDrivenDemoPolicy,
    PolicyEvaluator,
    call_policy,
    default_action_applier,
    default_observation_getter,
)

from .common import (
    ExampleLoopHooks,
    get_config_dir,
    parse_round_selection,
    prepare_task_file,
    print_final_summary,
    run_example_rounds,
)


@hydra.main(
    config_path=str(get_config_dir()),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # Validate the round selection before building the environment.
    rounds, selected_rounds = parse_round_selection(
        cfg.get("round_selection"), cfg.get("rounds")
    )
    task_file = prepare_task_file(cfg)
    policy_cfg = cfg.get("policy")
    policy = (
        instantiate(policy_cfg) if policy_cfg is not None else ConfigDrivenDemoPolicy()
    )
    action_applier = getattr(policy, "action_applier", default_action_applier)
    observation_getter = getattr(
        policy,
        "observation_getter",
        default_observation_getter,
    )
    max_updates_cfg = cfg.get("max_updates")
    max_updates = None if max_updates_cfg is None else int(max_updates_cfg)
    use_input = bool(cfg.get("use_input", False))
    get_obs = bool(cfg.get("get_obs", False))
    print_updates = bool(cfg.get("print_updates", True))

    evaluator = PolicyEvaluator(
        action_applier=action_applier,
        observation_getter=observation_getter,
    ).from_config(task_file)

    try:
        round_summaries = run_example_rounds(
            rounds=rounds,
            use_input=use_input,
            selected_rounds=selected_rounds,
            hooks=ExampleLoopHooks(
                reset_fn=evaluator.reset,
                step_fn=lambda _step, update: evaluator.update(
                    call_policy(
                        policy,
                        evaluator.get_observation() if get_obs else {},
                        update,
                        evaluator,
                    )
                ),
                summarize_fn=lambda update, steps_used, max_updates, elapsed_time_sec: (
                    evaluator.summarize(
                        update,
                        max_updates=max_updates,
                        updates_used=steps_used,
                        elapsed_time_sec=elapsed_time_sec,
                    )
                ),
                records_fn=lambda: evaluator.records,
                update_limit_fn=evaluator.terminate_unfinished_at_update_limit,
                before_round_fn=lambda _r: getattr(policy, "reset", lambda: None)(),
                reset_label="Reset evaluator",
                start_label="Starting policy rollout...",
                max_updates=max_updates,
                print_updates=print_updates,
                quiet_context_fn=evaluator.defer_viewer_updates,
            ),
        )
    finally:
        evaluator.close()

    print_final_summary(round_summaries)


if __name__ == "__main__":
    main()
