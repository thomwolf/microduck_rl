from scripts.eval_head_spin import _percentile, _summarize


def test_percentile_uses_nearest_rank():
    assert _percentile([4.0, 1.0, 3.0, 2.0], 0.75) == 3.0


def test_summary_reports_stage_rates_and_motion_limits():
    records = [
        {
            "entry": True,
            "turn_complete": True,
            "earned_turn": True,
            "stable_success": True,
            "timeout": False,
            "progress_fraction": 1.0,
            "episode_s": 3.0,
            "max_planar_drift_m": 0.02,
            "entry_planar_drift_m": 0.01,
            "turn_planar_drift_m": 0.015,
            "final_planar_drift_m": 0.02,
            "supported_head_drift_m": 0.01,
            "peak_abs_yaw_rate_rad_s": 3.0,
            "height_criterion": True,
            "upright_criterion": True,
            "both_feet_criterion": True,
            "head_clear_criterion": True,
            "standing_pose_criterion": True,
            "pose_and_linear_criterion": True,
            "pose_and_angular_criterion": True,
            "instant_stable_criterion": True,
            "best_stable_hold_s": 0.4,
        },
        {
            "entry": True,
            "turn_complete": False,
            "earned_turn": False,
            "stable_success": False,
            "timeout": True,
            "progress_fraction": 0.5,
            "episode_s": 7.0,
            "max_planar_drift_m": 0.08,
            "entry_planar_drift_m": 0.03,
            "turn_planar_drift_m": 0.06,
            "final_planar_drift_m": 0.07,
            "supported_head_drift_m": 0.04,
            "peak_abs_yaw_rate_rad_s": 5.0,
            "height_criterion": False,
            "upright_criterion": True,
            "both_feet_criterion": False,
            "head_clear_criterion": True,
            "standing_pose_criterion": False,
            "pose_and_linear_criterion": False,
            "pose_and_angular_criterion": False,
            "instant_stable_criterion": False,
            "best_stable_hold_s": 0.0,
        },
    ]

    summary = _summarize(records)

    assert summary["entry_rate"] == 1.0
    assert summary["earned_turn_rate"] == 0.5
    assert summary["stable_success_rate"] == 0.5
    assert summary["mean_progress_fraction"] == 0.75
    assert summary["p95_planar_drift_m"] == 0.08
    assert summary["p95_entry_planar_drift_m"] == 0.03
    assert summary["p95_turn_planar_drift_m"] == 0.06
    assert summary["p95_final_planar_drift_m"] == 0.07
    assert summary["p95_supported_head_drift_m"] == 0.04
    assert summary["upright_criterion_rate"] == 1.0
    assert summary["standing_pose_criterion_rate"] == 0.5
    assert summary["p95_best_stable_hold_s"] == 0.4
