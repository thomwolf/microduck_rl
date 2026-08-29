import math
from types import SimpleNamespace

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
    assert "head_spin_pivot" not in cfg.rewards
    assert cfg.rewards["head_spin_wrong_direction"].weight < 0.0
    assert cfg.rewards["head_spin_planar_drift"].weight < 0.0


def test_recovery_is_hard_gated_at_the_full_target_angle():
    cfg = make_microduck_head_spin_env_cfg()
    landing = cfg.rewards["head_spin_stability_score"]
    assert landing.params["target_angle"] == math.pi
    assert "gate_lo" not in landing.params
    assert "head_spin_upright_after_turn" not in cfg.rewards
    assert "head_spin_height_after_turn" not in cfg.rewards
    assert "head_spin_landing_sharp" not in cfg.rewards
    assert "head_spin_rise_velocity" not in cfg.rewards


def test_stability_score_is_maximal_only_for_a_quiet_complete_stand():
    perfect = mdp.head_spin_stability_score_from_components(
        cos_tilt=torch.tensor([1.0]),
        height=torch.tensor([0.11]),
        both_feet=torch.tensor([True]),
        head_clear=torch.tensor([True]),
        linear_speed=torch.tensor([0.0]),
        angular_speed=torch.tensor([0.0]),
        target_height=0.11,
    )
    inverted = mdp.head_spin_stability_score_from_components(
        cos_tilt=torch.tensor([-1.0]),
        height=torch.tensor([0.11]),
        both_feet=torch.tensor([True]),
        head_clear=torch.tensor([True]),
        linear_speed=torch.tensor([0.0]),
        angular_speed=torch.tensor([0.0]),
        target_height=0.11,
    )
    moving = mdp.head_spin_stability_score_from_components(
        cos_tilt=torch.tensor([1.0]),
        height=torch.tensor([0.11]),
        both_feet=torch.tensor([True]),
        head_clear=torch.tensor([True]),
        linear_speed=torch.tensor([0.30]),
        angular_speed=torch.tensor([2.0]),
        target_height=0.11,
    )
    assert torch.allclose(perfect, torch.ones(1))
    assert torch.allclose(inverted, torch.zeros(1))
    assert 0.2 < moving.item() < perfect.item()


def test_compactness_cost_applies_in_every_phase_and_is_stronger_after_turn():
    velocity = torch.tensor([[0.3, 0.4], [0.3, 0.4], [0.3, 0.4]])
    cost = mdp.head_spin_planar_motion_cost_from_components(
        velocity,
        supported=torch.tensor([False, True, False]),
        complete=torch.tensor([False, False, True]),
        supported_scale=4.0,
        post_turn_scale=4.0,
    )
    assert torch.allclose(cost, torch.tensor([0.25, 1.0, 1.0]))


def test_compactness_score_directly_tracks_maximum_drift():
    drift = torch.tensor([0.0, 0.15, 0.30, float("nan")])
    score = mdp.head_spin_compactness_score_from_drift(drift, drift_scale=0.15)
    expected = torch.tensor([1.0, math.exp(-1.0), math.exp(-4.0)])
    assert torch.allclose(score[:3], expected)
    assert score[3] == 0.0


def test_phase_motion_cost_uses_head_only_during_supported_yaw():
    root = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    head = torch.tensor([[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]])
    cost = mdp.head_spin_phase_planar_motion_cost_from_components(
        root,
        head,
        supported=torch.tensor([False, True, True]),
        complete=torch.tensor([False, False, True]),
        supported_scale=8.0,
        post_turn_scale=8.0,
    )
    assert torch.allclose(cost, torch.tensor([1.0, 2.0, 8.0]))


def test_cfg_polish_scales_match_strict_stability_thresholds():
    cfg = make_microduck_head_spin_env_cfg()
    stability = cfg.rewards["head_spin_stability_score"]
    drift = cfg.rewards["head_spin_planar_drift"]
    assert stability.params["linear_speed_scale"] == 0.15
    assert stability.params["angular_speed_scale"] == 1.0
    assert drift.params["supported_scale"] == 8.0
    assert drift.params["post_turn_scale"] == 8.0
    assert cfg.rewards["head_spin_planar_displacement"].weight == -100.0
    assert cfg.rewards["head_spin_head_pivot_displacement"].weight == -20.0
    compact = cfg.rewards["head_spin_compact_success"]
    assert compact.weight == 60.0
    assert compact.params["drift_scale"] == 0.20


def test_hard_completion_gate_does_not_leak_before_pi():
    env = SimpleNamespace(
        _head_spin_accum=torch.tensor([math.radians(179.0), math.pi]),
        _head_spin_max=torch.tensor([math.radians(179.0), math.pi]),
        _head_spin_paid=torch.zeros(2),
        _head_spin_head_latch=torch.ones(2, dtype=torch.bool),
    )
    assert torch.equal(
        mdp._head_spin_complete(env, math.pi),
        torch.tensor([False, True]),
    )


def test_potential_rate_is_frequency_independent_after_reward_dt_scaling():
    current = torch.tensor([0.7, 0.2])
    previous = torch.tensor([0.5, 0.4])
    active = torch.tensor([True, False])
    dt = 0.02
    rate = mdp._head_spin_potential_rate(current, previous, active, dt)
    assert torch.allclose(rate * dt, torch.tensor([0.2, 0.0]))


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


def test_cfg_requires_a_stable_success_hold_and_terminates_on_success():
    cfg = make_microduck_head_spin_env_cfg()
    success = cfg.rewards["head_spin_stable_success"]
    termination = cfg.terminations["head_spin_success"]
    assert success.params["hold_s"] == 0.4
    assert termination.params["hold_s"] == 0.4
    assert termination.time_out is False


def test_reverse_curriculum_starts_with_all_three_stages():
    cfg = make_microduck_head_spin_env_cfg()
    reset = cfg.events["set_head_spin_state"].params
    assert reset["standing_prob"] > 0.0
    assert reset["headstand_prob"] > 0.0
    assert reset["recovery_prob"] > 0.0
    assert reset["spawn_progress_range"][1] < HEAD_SPIN_TARGET_ANGLE
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
    assert MicroduckHeadSpinRlCfg.algorithm.gamma == 0.995
