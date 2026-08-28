import math

import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_head_spin_env_cfg import (
    HEAD_SPIN_TARGET_ANGLE,
    MicroduckHeadSpinRlCfg,
    make_microduck_head_spin_env_cfg,
)


def test_yaw_delta_is_signed_and_requires_valid_head_support():
    omega = torch.tensor([2.0, -2.0, 2.0])
    support = torch.tensor([True, True, False])
    delta = mdp.head_spin_yaw_delta(omega, support, step_dt=0.1, direction=1.0)
    assert torch.allclose(delta, torch.tensor([0.2, -0.2, 0.0]))


def test_yaw_delta_flips_with_requested_direction():
    omega = torch.tensor([2.0, -2.0])
    support = torch.ones(2, dtype=torch.bool)
    delta = mdp.head_spin_yaw_delta(omega, support, step_dt=0.1, direction=-1.0)
    assert torch.allclose(delta, torch.tensor([-0.2, 0.2]))


def test_stage_observation_switches_enter_spin_recover_at_pi():
    latched = torch.tensor([False, True, True])
    max_yaw = torch.tensor([0.0, math.pi - 0.01, math.pi])
    obs = mdp.head_spin_stage_from_state(latched, max_yaw, direction=1.0)
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    assert torch.equal(obs, expected)


def test_cfg_requires_a_supported_half_turn_and_has_no_yaw_blocker():
    cfg = make_microduck_head_spin_env_cfg()
    progress = cfg.rewards["head_spin_progress"]
    assert progress.params["target_angle"] == math.pi
    assert progress.weight > 0.0
    assert "angular_momentum" not in cfg.rewards
    assert "upright" not in cfg.rewards
    assert cfg.rewards["head_spin_wrong_direction"].weight < 0.0
    assert cfg.rewards["head_spin_planar_drift"].weight < 0.0


def test_cfg_uses_head_and_feet_contact_sensors():
    cfg = make_microduck_head_spin_env_cfg()
    names = {sensor.name for sensor in cfg.scene.sensors}
    assert "head_ground_contact" in names
    assert "feet_ground_contact" in names


def test_cfg_exposes_stagewise_training_metrics():
    cfg = make_microduck_head_spin_env_cfg()
    for name in (
        "head_spin_progress",
        "head_spin_turn_complete",
        "head_spin_support",
        "head_spin_full_success",
        "head_spin_recovery_success",
    ):
        assert name in cfg.metrics
    assert cfg.metrics["head_spin_turn_complete"].reduce == "last"
    assert cfg.metrics["head_spin_full_success"].reduce == "last"
    assert cfg.metrics["head_spin_recovery_success"].reduce == "last"


def test_reverse_curriculum_starts_with_all_three_stages():
    cfg = make_microduck_head_spin_env_cfg()
    reset = cfg.events["set_head_spin_state"].params
    assert reset["standing_prob"] > 0.0
    assert reset["headstand_prob"] > 0.0
    assert reset["recovery_prob"] > 0.0
    assert (
        sum(
            reset[name] for name in ("standing_prob", "headstand_prob", "recovery_prob")
        )
        == 1.0
    )


def test_play_cfg_always_starts_standing():
    cfg = make_microduck_head_spin_env_cfg(play=True)
    reset = cfg.events["set_head_spin_state"].params
    assert reset["standing_prob"] == 1.0
    assert reset["headstand_prob"] == 0.0
    assert reset["recovery_prob"] == 0.0


def test_actor_observation_keeps_shared_layout_and_uses_stage_slot():
    from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
        make_microduck_roulade_env_cfg,
    )

    head_spin = make_microduck_head_spin_env_cfg()
    roulade = make_microduck_roulade_env_cfg()
    for group in ("actor", "critic"):
        assert list(head_spin.observations[group].terms) == list(
            roulade.observations[group].terms
        )
        stage = head_spin.observations[group].terms["head_command"]
        assert stage.func is mdp.head_spin_stage_observation


def test_fixed_direction_disables_symmetry_augmentation():
    assert HEAD_SPIN_TARGET_ANGLE == math.pi
    assert MicroduckHeadSpinRlCfg.algorithm.symmetry_cfg is None
