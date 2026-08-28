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
| `headspin-rv2-seed42-stage1` | `6a91f596984507d9db4ea699` | Reward-v2 baseline through checkpoint 500 | Canceled after recovery stalled | <= $0.55 |
| `headspin-rv3-seed42-recovery` | `6a91fe94984507d9db4ea75d` | Checkpoint-500 recovery-focused warm start | Running; 2 h hard timeout | <= $1.20 exposure |
| **Completed/canceled cumulative estimate** | | | | **$0.75** |

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
- Status: canceled after checkpoint 500; superseded by Experiment 2

Checkpoint 250 was evaluated locally with 64 held-out episodes per bucket
(seed 25042). Entry and turn completion were 100% in every spin bucket,
including 100% full entry-plus-turn from standing. Stable success was 0% in
the four turn buckets and 1/64 (1.5625%) in the recovery bucket. The standing
bucket had p95 yaw rate 8.14 rad/s and p95 drift 0.216 m; the isolated spin
buckets were worse. This localizes the next learning problem to braking and
recovery, while the nonzero recovery success proves the strict terminal state
is reachable. Keep the baseline unchanged through checkpoint 500 before
deciding whether to rebalance the spawn curriculum.

Checkpoint 500 was evaluated with the identical 64 episodes per bucket and
seed 25042. Entry and turn completion remained 100%, but stable success was
0% in every bucket, including recovery. Standing p95 yaw rate regressed from
8.14 to 9.25 rad/s and drift from 0.216 to 0.259 m. This rejects the hypothesis
that the original 20% recovery sampling was producing reliable recovery by
iteration 500. Stop the baseline and warm-start checkpoint 500 with reward
weights unchanged but a 10% standing / 20% headstand / 70% recovery spawn mix.
That isolates curriculum allocation as the next experimental variable.

## Experiment 2: checkpoint-500 recovery focus

Warm-start `model_500.pt` from Experiment 1, preserve PPO optimizer and
observation normalizer state, and train for 1,000 additional iterations. The
only task change is the spawn schedule: 10/20/70 standing/headstand/recovery at
the first stage, then 20/30/50, 40/35/25, and finally 60/30/10 as later stages
are crossed. Promote only if recovery stable success rises materially without
losing the already-solved entry and 180-degree turn.

HF job: `6a91fe94984507d9db4ea75d`. The bootstrap explicitly staged and loaded
`model_500.pt`; the first logged learning iteration was 500/1500 and the live
spawn mix was 10/20/70 as intended. Experiment 1 was canceled only after this
load was confirmed.
