"""Record one deterministic standing-start head-spin policy rollout."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

import mjlab_microduck.tasks  # noqa: F401  # populate the task registry


TASK_ID = "Mjlab-HeadSpin-Flat-MicroDuck"


def main() -> int:
    """Record one episode and require the strict stable-success termination."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=0.8)
    parser.add_argument("--camera-elevation", type=float, default=-10.0)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("head_spin_rollout.mp4"))
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing video: {args.output}")

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.auto_reset = False
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    env_cfg.viewer.distance = args.camera_distance
    env_cfg.viewer.elevation = args.camera_elevation
    env_cfg.viewer.azimuth = args.camera_azimuth
    agent_cfg = load_rl_cfg(args.task)
    agent_cfg.seed = args.seed

    base = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    output_dir = args.output.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output.stem
    recorder = VideoRecorder(
        base,
        video_folder=output_dir,
        step_trigger=lambda step: step == 0,
        video_length=None,
        name_prefix=prefix,
        disable_logger=False,
    )
    env = RslRlVecEnvWrapper(recorder, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(args.checkpoint_file),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    all_ids = torch.arange(base.num_envs, device=base.device, dtype=torch.long)
    base.reset(env_ids=all_ids)
    observations = env.get_observations()
    done = False
    with torch.no_grad():
        for _ in range(args.max_steps):
            actions = policy(observations)
            observations, _, dones, _ = env.step(actions)
            if bool(dones[0].item()):
                done = True
                break

    success = done and bool(base._head_spin_success_paid[0].item())
    env.close()
    generated = output_dir / f"{prefix}-step-0.mp4"
    if not generated.exists():
        raise RuntimeError("The rollout completed without producing a video")
    generated.rename(args.output.resolve())
    print(f"[record] strict_success={success} video={args.output.resolve()}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
