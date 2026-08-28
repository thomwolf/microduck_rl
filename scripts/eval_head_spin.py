"""Deterministic, stage-conditioned evaluation for MicroDuck head-spin policies."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import mjlab_microduck.tasks  # noqa: F401  # populate the task registry


TASK_ID = "Mjlab-HeadSpin-Flat-MicroDuck"
TARGET_ANGLE = math.pi

SCENARIOS: dict[str, dict[str, object]] = {
    "standing_full": {
        "standing_prob": 1.0,
        "headstand_prob": 0.0,
        "recovery_prob": 0.0,
        "spawn_progress_range": (0.0, 0.0),
        "initial_spin_rate_range": (0.0, 0.0),
    },
    "spin_from_0deg": {
        "standing_prob": 0.0,
        "headstand_prob": 1.0,
        "recovery_prob": 0.0,
        "spawn_progress_range": (0.0, 0.0),
    },
    "spin_from_90deg": {
        "standing_prob": 0.0,
        "headstand_prob": 1.0,
        "recovery_prob": 0.0,
        "spawn_progress_range": (math.pi / 2.0, math.pi / 2.0),
    },
    "spin_from_160deg": {
        "standing_prob": 0.0,
        "headstand_prob": 1.0,
        "recovery_prob": 0.0,
        "spawn_progress_range": (math.radians(160.0), math.radians(160.0)),
    },
    "recovery": {
        "standing_prob": 0.0,
        "headstand_prob": 0.0,
        "recovery_prob": 1.0,
        "spawn_progress_range": (TARGET_ANGLE, TARGET_ANGLE),
        "initial_spin_rate_range": (0.0, 0.0),
    },
}


def _percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile without an optional dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(index, 0)]


def _summarize(records: list[dict[str, float | bool]]) -> dict[str, float | int]:
    """Aggregate episode records into promotion metrics."""
    count = len(records)

    def rate(key: str) -> float:
        return sum(bool(record[key]) for record in records) / max(count, 1)

    return {
        "episodes": count,
        "entry_rate": rate("entry"),
        "turn_complete_rate": rate("turn_complete"),
        "earned_turn_rate": rate("earned_turn"),
        "stable_success_rate": rate("stable_success"),
        "timeout_rate": rate("timeout"),
        "mean_progress_fraction": fmean(
            float(record["progress_fraction"]) for record in records
        ),
        "mean_episode_s": fmean(float(record["episode_s"]) for record in records),
        "p95_planar_drift_m": _percentile(
            [float(record["max_planar_drift_m"]) for record in records], 0.95
        ),
        "p95_abs_yaw_rate_rad_s": _percentile(
            [float(record["peak_abs_yaw_rate_rad_s"]) for record in records], 0.95
        ),
    }


def _evaluate_scenario(
    env: RslRlVecEnvWrapper,
    policy,
    name: str,
    reset_params: dict[str, object],
    episodes: int,
) -> dict[str, float | int]:
    """Evaluate one reset bucket with asynchronous manual resets."""
    base = env.unwrapped
    event_cfg = base.event_manager.get_term_cfg("set_head_spin_state")
    event_cfg.params.update(reset_params)

    all_ids = torch.arange(base.num_envs, device=base.device, dtype=torch.long)
    base.reset(env_ids=all_ids)
    observations = env.get_observations()
    asset = base.scene["robot"]
    start_xy = asset.data.root_link_pos_w[:, :2].clone()
    max_drift = torch.zeros(base.num_envs, device=base.device)
    peak_yaw_rate = torch.zeros(base.num_envs, device=base.device)
    records: list[dict[str, float | bool]] = []

    while len(records) < episodes:
        # The environment mutates actuator delay buffers in-place; inference
        # tensors cannot later be reset outside InferenceMode. no_grad keeps
        # policy evaluation cheap without changing the buffers' tensor type.
        with torch.no_grad():
            actions = policy(observations)
            observations, _, dones, _ = env.step(actions)

        displacement = torch.linalg.vector_norm(
            asset.data.root_link_pos_w[:, :2] - start_xy, dim=1
        )
        max_drift = torch.maximum(max_drift, displacement)
        peak_yaw_rate = torch.maximum(
            peak_yaw_rate, asset.data.root_link_ang_vel_w[:, 2].abs()
        )

        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) == 0:
            continue

        for env_id in done_ids.tolist():
            if len(records) >= episodes:
                break
            max_yaw = float(base._head_spin_max[env_id].item())
            entry = bool(base._head_spin_head_latch[env_id].item())
            seeded_complete = bool(base._head_spin_spawned_complete[env_id].item())
            stable_success = bool(base._head_spin_success_paid[env_id].item())
            records.append(
                {
                    "entry": entry,
                    "turn_complete": entry and max_yaw >= TARGET_ANGLE,
                    "earned_turn": (
                        entry and max_yaw >= TARGET_ANGLE and not seeded_complete
                    ),
                    "stable_success": stable_success,
                    "timeout": bool(base.reset_time_outs[env_id].item()),
                    "progress_fraction": min(max_yaw / TARGET_ANGLE, 1.0),
                    "episode_s": float(base.episode_length_buf[env_id].item())
                    * base.step_dt,
                    "max_planar_drift_m": float(max_drift[env_id].item()),
                    "peak_abs_yaw_rate_rad_s": float(peak_yaw_rate[env_id].item()),
                }
            )

        base.reset(env_ids=done_ids)
        observations = env.get_observations()
        start_xy[done_ids] = asset.data.root_link_pos_w[done_ids, :2]
        max_drift[done_ids] = 0.0
        peak_yaw_rate[done_ids] = 0.0

    summary = _summarize(records)
    print(f"[eval] {name}: {json.dumps(summary, sort_keys=True)}", flush=True)
    return summary


def main() -> int:
    """Run all evaluation buckets and write a machine-readable report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--episodes-per-scenario", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("head_spin_eval.json"))
    args = parser.parse_args()

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.auto_reset = False
    agent_cfg = load_rl_cfg(args.task)
    agent_cfg.seed = args.seed

    base = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(args.checkpoint_file),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    results = {
        name: _evaluate_scenario(
            env,
            policy,
            name,
            params,
            args.episodes_per_scenario,
        )
        for name, params in SCENARIOS.items()
    }
    report = {
        "task": args.task,
        "checkpoint": str(args.checkpoint_file),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "episodes_per_scenario": args.episodes_per_scenario,
        "scenarios": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[eval] report -> {args.output}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
