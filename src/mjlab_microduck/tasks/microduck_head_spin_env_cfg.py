"""Microduck episodic head-spin task.

The policy must leave a stand, establish flat top-of-head support with both
feet clear, rotate at least 180 degrees around world vertical in one requested
direction, and recover to a stable stand. Reward gates are state-based; the
4D head-command observation slot carries enter/spin/recover state so a
feed-forward policy can know when an axially symmetric half-turn is complete.
"""

import math
from copy import deepcopy

from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    MetricsTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    STAND_Z,
    TUCK_OVERRIDES,
    _LEG_JOINTS,
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
)


EPISODE_LENGTH_S = 7.0
HEAD_SPIN_TARGET_ANGLE = math.pi
HEAD_SPIN_DIRECTION = 1.0
HEAD_SPIN_SUCCESS_HOLD_S = 0.4


def make_microduck_head_spin_env_cfg(play: bool = False):
    """Create the flat-ground Microduck head-spin environment configuration."""
    # Roulade is the repository's validated dynamic-maneuver base: it already
    # supplies the all-collisions robot, head/feet/whole-body contact sensors,
    # 61D observation ordering, actuator randomization, and late regularizers.
    cfg = make_microduck_roulade_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    for name in tuple(cfg.rewards):
        if name.startswith("roulade_"):
            cfg.rewards.pop(name)

    # Stage 1: potential progress toward the headstand plus a fixed event bonus
    # for establishing valid top-of-head support. Both are unfarmable.
    cfg.rewards["head_spin_entry_progress"] = RewardTermCfg(
        func=microduck_mdp.head_spin_entry_progress,
        weight=2.0,
        params={"direction": HEAD_SPIN_DIRECTION},
    )
    cfg.rewards["head_spin_support_entry"] = RewardTermCfg(
        func=microduck_mdp.head_spin_support_entry,
        weight=2.0,
        params={"direction": HEAD_SPIN_DIRECTION},
    )

    # Stage 2: the sole positive spin signal is new signed supported-yaw
    # frontier. There is no contact/rate annuity to farm after completion.
    cfg.rewards["head_spin_progress"] = RewardTermCfg(
        func=microduck_mdp.head_spin_progress,
        weight=8.0,
        params={
            "target_angle": HEAD_SPIN_TARGET_ANGLE,
            "max_paid_rate": 4.0,
            "direction": HEAD_SPIN_DIRECTION,
        },
    )
    # Stage 3 is HARD-gated at pi. Recovery progress pays changes in a bounded
    # potential; the small composite annuity teaches the 0.4 s stable hold; the
    # one-shot success event dominates and terminates the completed episode.
    cfg.rewards["head_spin_recovery_progress"] = RewardTermCfg(
        func=microduck_mdp.head_spin_recovery_progress,
        weight=5.0,
        params={
            "target_height": STAND_Z,
            "target_angle": HEAD_SPIN_TARGET_ANGLE,
            "direction": HEAD_SPIN_DIRECTION,
        },
    )
    cfg.rewards["head_spin_landing_composite"] = RewardTermCfg(
        func=microduck_mdp.head_spin_landing_composite,
        weight=1.0,
        params={
            "target_height": STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "joint_indices": _LEG_JOINTS,
            "target_angle": HEAD_SPIN_TARGET_ANGLE,
            "target_overrides": None,
            "direction": HEAD_SPIN_DIRECTION,
        },
    )
    cfg.rewards["head_spin_stable_success"] = RewardTermCfg(
        func=microduck_mdp.head_spin_stable_success_bonus,
        weight=12.0,
        params={
            "target_angle": HEAD_SPIN_TARGET_ANGLE,
            "direction": HEAD_SPIN_DIRECTION,
            "hold_s": HEAD_SPIN_SUCCESS_HOLD_S,
        },
    )
    cfg.rewards["head_spin_turn_brake"] = RewardTermCfg(
        func=microduck_mdp.head_spin_turn_brake_penalty,
        weight=-0.1,
        params={
            "target_angle": HEAD_SPIN_TARGET_ANGLE,
            "direction": HEAD_SPIN_DIRECTION,
        },
    )

    cfg.rewards["head_spin_wrong_direction"] = RewardTermCfg(
        func=microduck_mdp.head_spin_wrong_direction_penalty,
        weight=-0.2,
        params={"direction": HEAD_SPIN_DIRECTION},
    )
    cfg.rewards["head_spin_overspeed"] = RewardTermCfg(
        func=microduck_mdp.head_spin_overspeed_penalty,
        weight=-0.05,
        params={"omega_max": 6.0},
    )
    cfg.rewards["head_spin_planar_drift"] = RewardTermCfg(
        func=microduck_mdp.head_spin_planar_drift_penalty,
        weight=-1.0,
    )

    # Yaw is the task, so the 3D angular-momentum regularizer must not oppose
    # it. body_ang_vel remains tiny and only penalizes world x/y rotation.
    cfg.rewards.pop("angular_momentum", None)

    cfg.terminations["head_spin_success"] = TerminationTermCfg(
        func=microduck_mdp.head_spin_stable_success_termination,
        time_out=False,
        params={
            "target_angle": HEAD_SPIN_TARGET_ANGLE,
            "direction": HEAD_SPIN_DIRECTION,
            "hold_s": HEAD_SPIN_SUCCESS_HOLD_S,
        },
    )

    # Preserve the shared 61D layout while giving a feed-forward policy the
    # missing task state: [enter, spin, recover, direction]. Absolute yaw is
    # deliberately absent from proprioception, so zeros here would make the
    # completion transition unobservable.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.head_spin_stage_observation,
            params={
                "target_angle": HEAD_SPIN_TARGET_ANGLE,
                "direction": HEAD_SPIN_DIRECTION,
            },
        )

    command = cfg.commands["twist"]
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2.0)

    cfg.events.pop("set_roulade_state", None)
    reset_params = {
        "standing_prob": 1.0 if play else 0.25,
        "headstand_prob": 0.0 if play else 0.55,
        "recovery_prob": 0.0 if play else 0.20,
        "standing_z_range": (0.11, 0.12),
        "standing_tilt_max": math.radians(5.0),
        "headstand_pitch_range": (math.radians(95.0), math.radians(125.0)),
        "headstand_z_range": (0.05, 0.085),
        "headstand_roll_max": math.radians(4.0),
        "initial_spin_rate_range": (0.0, 2.0),
        "target_angle": HEAD_SPIN_TARGET_ANGLE,
        "spawn_progress_range": (0.0, math.radians(160.0)),
        "direction": HEAD_SPIN_DIRECTION,
        "tuck_overrides": TUCK_OVERRIDES,
        "tuck_factor_range": (0.7, 1.0),
        "joint_noise_std": 0.05,
    }
    cfg.events["set_head_spin_state"] = EventTermCfg(
        func=microduck_mdp.reset_head_spin_state,
        mode="reset",
        params=reset_params,
    )

    cfg.curriculum.pop("roulade_spawn_mix", None)
    if not play:
        cfg.curriculum["head_spin_spawn_mix"] = CurriculumTermCfg(
            func=microduck_mdp.event_param_curriculum,
            params={
                "event_name": "set_head_spin_state",
                "param_stages": [
                    {
                        "step": 0,
                        "params": {
                            "standing_prob": 0.25,
                            "headstand_prob": 0.55,
                            "recovery_prob": 0.20,
                        },
                    },
                    {
                        "step": 3000 * 24,
                        "params": {
                            "standing_prob": 0.40,
                            "headstand_prob": 0.45,
                            "recovery_prob": 0.15,
                        },
                    },
                    {
                        "step": 6000 * 24,
                        "params": {
                            "standing_prob": 0.60,
                            "headstand_prob": 0.30,
                            "recovery_prob": 0.10,
                        },
                    },
                ],
            },
        )

    cfg.metrics["head_spin_progress"] = MetricsTermCfg(
        func=microduck_mdp.head_spin_progress_fraction,
        params={
            "target_angle": HEAD_SPIN_TARGET_ANGLE,
            "direction": HEAD_SPIN_DIRECTION,
        },
        reduce="last",
    )
    cfg.metrics["head_spin_turn_complete"] = MetricsTermCfg(
        func=microduck_mdp.head_spin_completion_metric,
        params={
            "target_angle": HEAD_SPIN_TARGET_ANGLE,
            "direction": HEAD_SPIN_DIRECTION,
        },
        reduce="last",
    )
    cfg.metrics["head_spin_support"] = MetricsTermCfg(
        func=microduck_mdp.head_spin_support_metric,
    )
    cfg.metrics["head_spin_full_success"] = MetricsTermCfg(
        func=microduck_mdp.head_spin_final_stand_metric,
        params={
            "target_angle": HEAD_SPIN_TARGET_ANGLE,
            "spawn_bucket": "standing",
            "direction": HEAD_SPIN_DIRECTION,
            "hold_s": HEAD_SPIN_SUCCESS_HOLD_S,
        },
        reduce="last",
    )
    cfg.metrics["head_spin_recovery_success"] = MetricsTermCfg(
        func=microduck_mdp.head_spin_final_stand_metric,
        params={
            "target_angle": HEAD_SPIN_TARGET_ANGLE,
            "spawn_bucket": "recovery",
            "direction": HEAD_SPIN_DIRECTION,
            "hold_s": HEAD_SPIN_SUCCESS_HOLD_S,
        },
        reduce="last",
    )

    return cfg


MicroduckHeadSpinRlCfg = deepcopy(MicroduckRouladeRlCfg)
MicroduckHeadSpinRlCfg.algorithm.symmetry_cfg = None
MicroduckHeadSpinRlCfg.algorithm.gamma = 0.995
MicroduckHeadSpinRlCfg.experiment_name = "microduck_head_spin"
MicroduckHeadSpinRlCfg.run_name = "microduck_head_spin"
