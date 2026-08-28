# Head-spin experiment ledger

This file is the source of truth for head-spin training decisions, evaluation
results, and Hugging Face Jobs spend. One `l4x1` costs $0.80/hour, billed by the
minute while starting or running. The project must stop for approval before the
cumulative estimate can exceed $200.

## Promotion criteria

A checkpoint is evaluated deterministically in five reset buckets: standing,
headstand at 0°, 90°, and 160° progress, and turn-complete recovery. Each bucket
reports entry, turn, stable-success, timeout, drift, and peak yaw-rate metrics.

Development promotion gates:

1. recovery stable success >= 80%;
2. spin-from-160° stable success >= 70%;
3. spin-from-90° stable success >= 60%;
4. spin-from-0° stable success >= 50%;
5. standing full-sequence success >= 30% before robustification.

The final candidate must exceed 80% standing full-sequence success over at
least 1,000 held-out episodes per seed, across three seeds, with p95 planar
drift below 0.15 m and no NaN terminations. These are promotion gates, not
claims about the current policy.

## Cost ledger

| Run | Job | Purpose | Result | Estimated cost |
|---|---|---|---|---:|
| `headspin-smoke` | `6a91e04a45686a1580c1271a` | Initial pipeline smoke | Failed at W&B login | <= $0.05 |
| `headspin-smoke-v2` | `6a91e27745686a1580c12779` | Corrected 5-iteration smoke | Completed | <= $0.05 |
| `headspin-rv2-smoke` | `6a91ef23984507d9db4ea5c4` | Reward-v2 L4 pipeline smoke | Canceled while queued; no runtime | $0.00 |
| `headspin-rv2-smoke-t4` | `6a91f31b984507d9db4ea655` | Reward-v2 train/eval/export/upload smoke | Completed | <= $0.10 |
| **Cumulative conservative estimate** | | | | **$0.20** |

## Experiment 0: pipeline smoke

The five-iteration checkpoint was evaluated only to validate the evaluator. As
expected, it achieved zero stable success in all buckets and cannot be used to
judge the task design.

## Reward-v2 hypothesis

- Entry and recovery use bounded potential differences, not state annuities.
- The only positive spin signal is new signed supported-yaw frontier.
- Recovery remains completely locked until the supported frontier reaches pi.
- A stable success requires both feet, no head contact, bounded linear/angular
  speed, standing height and tilt, held continuously for 0.4 seconds.
- Stable success pays once and terminates the episode.
- PPO gamma is 0.995 so late success can credit earlier actions over the
  seven-second maneuver.

The first training experiment should test this reward unchanged. Subsequent
changes are selected from the failed evaluation bucket rather than total reward.

## Experiment 1: reward-v2 seed 42, stage 1

- HF job: `6a91f596984507d9db4ea699`
- Model repo: `pollen-robotics/headspin-rv2-seed42-stage1`
- Training: 2,048 environments, 3,000 iterations, seed 42, T4 medium
- Status: running

Checkpoint 250 was evaluated locally with 64 held-out episodes per bucket
(seed 25042). Entry and turn completion were 100% in every spin bucket,
including 100% full entry-plus-turn from standing. Stable success was 0% in
the four turn buckets and 1/64 (1.5625%) in the recovery bucket. The standing
bucket had p95 yaw rate 8.14 rad/s and p95 drift 0.216 m; the isolated spin
buckets were worse. This localizes the next learning problem to braking and
recovery, while the nonzero recovery success proves the strict terminal state
is reachable. Keep the baseline unchanged through checkpoint 500 before
deciding whether to rebalance the spawn curriculum.
