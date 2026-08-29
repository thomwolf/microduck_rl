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
least 1,000 held-out episodes per seed, across three seeds, with p95 planted-head
drift below 0.10 m, p95 final root displacement below 0.20 m, and no NaN
terminations. Maximum trunk excursion is still reported but is not a gate,
because the trunk must move around the planted head during inversion. These are
promotion gates, not claims about the current policy.

## Cost ledger

| Run | Job | Purpose | Result | Estimated cost |
|---|---|---|---|---:|
| `headspin-smoke` | `6a91e04a45686a1580c1271a` | Initial pipeline smoke | Failed at W&B login | <= $0.05 |
| `headspin-smoke-v2` | `6a91e27745686a1580c12779` | Corrected 5-iteration smoke | Completed | <= $0.05 |
| `headspin-rv2-smoke` | `6a91ef23984507d9db4ea5c4` | Reward-v2 L4 pipeline smoke | Canceled while queued; no runtime | $0.00 |
| `headspin-rv2-smoke-t4` | `6a91f31b984507d9db4ea655` | Reward-v2 train/eval/export/upload smoke | Completed | <= $0.10 |
| `headspin-rv2-seed42-stage1` | `6a91f596984507d9db4ea699` | Reward-v2 baseline through checkpoint 500 | Canceled after recovery stalled | <= $0.55 |
| `headspin-rv3-seed42-recovery` | `6a91fe94984507d9db4ea75d` | Checkpoint-500 recovery-focused warm start | Canceled after checkpoint 750 was evaluated | <= $0.40 |
| `headspin-rv4-seed42-stability` | `6a920658984507d9db4ea83f` | Checkpoint-750 success-aligned warm start | Completed at checkpoint 1249 | <= $0.35 |
| `headspin-rv5-seed42-compact` | `6a920f7a984507d9db4ea95b` | Compactness polish submission | Failed before training: obsolete seed flag | <= $0.05 |
| `headspin-rv5b-seed42-compact` | `6a921022984507d9db4ea973` | Corrected checkpoint-1249 compactness polish | Completed at checkpoint 1498 | <= $0.20 |
| `headspin-rv6-seed42-support-compact` | `6a92157645686a1580c1312f` | Checkpoint-1498 supported-spin compactness | Completed at checkpoint 1747 | <= $0.20 |
| `headspin-rv6b-seed42-support-compact` | `6a9219f4984507d9db4eaa9c` | Unchanged checkpoint-1747 continuation | Completed at checkpoint 1996 | <= $0.20 |
| `headspin-rv7-seed42-strong-compact` | `6a921ecd45686a1580c132b6` | Checkpoint-1996 8x compactness | Completed at checkpoint 2245 | <= $0.20 |
| `headspin-rv8-smoke-direct-drift` | `6a922473984507d9db4eab8a` | Direct-displacement integration smoke | Canceled after finite reward verification | <= $0.05 |
| `headspin-rv8-seed42-direct-drift` | `6a922576984507d9db4eaba9` | Checkpoint-2245 direct displacement shaping | Completed at checkpoint 2494 | <= $0.20 |
| `headspin-rv8-pivot-diagnostic` | `6a922a4045686a1580c134bf` | Evaluation-only head-pivot instrumentation | Evaluator API mismatch; no training | <= $0.05 |
| `headspin-rv8-pivot-diagnostic-v2` | `6a922b70984507d9db4eac27` | Corrected evaluation-only head-pivot instrumentation | Completed; no training | <= $0.05 |
| `headspin-rv9-smoke-phase-correct` | `6a922d12984507d9db4eac43` | Phase-correct reward integration smoke | Failed before first update: unresolved body slice | <= $0.05 |
| `headspin-rv9-smoke-phase-correct-v2` | `6a922e27984507d9db4eac64` | Corrected phase-correct reward smoke | Finite pivot reward; zero NaNs; canceled | <= $0.05 |
| **Completed/canceled cumulative estimate** | | | | **$2.80** |

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

Checkpoint 750 again achieved 100% entry and turn completion but 0% stable
success in every 64-episode bucket. A recovery-only component diagnostic found
that every episode reached standing height, both feet contact, and head release;
31.25% reached the 20-degree upright threshold, 17.19% reached the complete
standing pose, 7.81% combined pose with the linear-speed threshold, 6.25%
combined pose with the angular-speed threshold, and 1/64 reached all strict
criteria instantaneously. None held them for 0.4 s. Therefore increased recovery
sampling alone is insufficient: the learned exit reaches the feet but does not
arrest its residual tilt and motion.

## Experiment 3: success-aligned stability reward

Warm-start Experiment 2 checkpoint 750 with the same 10/20/70 spawn mix. Replace
the joint-HOME landing composite with a smooth post-turn score aligned to the
actual success predicates: upright trunk, standing height, both feet, released
head, low linear speed, and low angular speed. Keep the strict 0.4 s success
termination unchanged and raise its one-shot bonus from 12 to 30. The aligned
annuity has weight 4 and can pay at most 1.6 before the 0.4 s success termination,
so the terminal event still dominates partial-state farming. No spin or entry
reward changes are made.

HF job: `6a920658984507d9db4ea83f`. The bootstrap explicitly loaded Experiment 2
`model_750.pt`; training resumed at iteration 750/1250 with the new stability
score present in the live reward table. Experiment 2 was then canceled. Its
conservative runtime estimate is <= $0.40.

Checkpoint 1000 was evaluated locally on the same held-out seed 25042 with 64
episodes per bucket. Stable-success rates were 95.31% from standing, 42.19%
from 0-degree head support, 84.38% from 90 degrees, 90.63% from 160 degrees,
and 100% from turn-complete recovery. Every bucket retained 100% entry; turn
completion was 95.31% from 0 degrees and 100% elsewhere. This is the first
checkpoint to solve the strict 0.4-second stable hold and passes four of five
development gates. The remaining early-headstand gate is narrowly missed
(42.19% versus 50%), and p95 planar drift remains too high at 0.276--0.674 m.
Evaluate checkpoint 1250 before changing the reward; if drift persists, extend
the velocity penalty beyond head-only support rather than expecting a policy
without global-position observations to correct absolute displacement.

The final zero-based checkpoint 1249 achieved 100% strict stable success in all
five 256-episode buckets in the job's held-out evaluation (seed 20260828). A
second local 256-episode standing evaluation on seed 35042 also achieved 100%,
with no timeouts or missed criteria. This promotes reward-v4 as the immutable
success fallback. Standing p95 drift was 0.303 m and 0.302 m respectively,
however, and the headstand buckets drifted 0.507--0.741 m, so it does not pass
the final compactness gate.

## Experiment 4: compactness polish

Warm-start checkpoint 1249 without changing the 180-degree task gate, strict
success predicates, observations, or spawn curriculum. Make three targeted
changes based on the residual-motion evidence:

- set the dense stillness scales to the actual strict limits (0.15 m/s linear
  and 1.0 rad/s angular) and raise its weight from 4 to 6;
- raise the post-turn yaw-brake penalty from -0.1 to -0.25;
- apply squared planar-speed cost throughout the maneuver rather than only on
  valid head support, with a 4x multiplier after turn completion.

Checkpoint 1249 remains the fallback. Promote a polish checkpoint only if it
retains 100% development success while reducing standing p95 drift; reject it
immediately if strict success regresses materially.

The first submission failed before PPO started because the current trainer
expects `--agent.seed` rather than the obsolete `--seed` shorthand. The
corrected job `6a921022984507d9db4ea973` uses both agent and environment seed 42.
It explicitly loaded `model_1249.pt`, resumed at iteration 1249/1499, and its
live reward table showed stability 6.0, turn brake -0.25, and planar motion
-1.0. No NaN terminations appeared during initialization.

Checkpoint 1498 retained 100% strict success in every official 256-episode
bucket. It improved p95 drift in the head-start buckets but did not improve the
primary standing bucket: official p95 drift changed from 0.303 m to 0.311 m.
It is therefore not promoted over checkpoint 1249.

Phase-instrumented standing evaluation localized the fallback's p95 drift to
0.069 m at first head support, 0.223 m at turn completion, and 0.299 m at the
final stand. The matching checkpoint-1498 values were 0.063 m, 0.209 m, and
0.283 m. Most displacement is created during supported yaw; reward-v5 kept the
old -1 planar-speed weight in that phase and only extended it to entry and
post-turn recovery, explaining the limited improvement.

## Experiment 5: supported-turn compactness

Warm-start checkpoint 1498, which preserves 100% success, and keep all reward-v5
terms unchanged except for a 4x multiplier on planar-speed cost during valid
head-only support. Entry retains scale 1 and post-turn recovery retains scale 4.
This directly targets the phase that creates about three quarters of the final
displacement. Promote only if standing success remains 100% and both
turn-completion and final p95 drift improve materially.

HF job `6a92157645686a1580c1312f` uses source commit `a59bc24`. It explicitly
loaded `model_1498.pt` and resumed at iteration 1498/1748 with finite rewards,
2,048 environments, agent/environment seed 42, and no NaN terminations.

Checkpoint 1747 retained 100% strict success in every official 256-episode
bucket. Standing p95 drift improved to 0.257 m, with 0.188 m accumulated by
turn completion. A second 256-episode standing batch on seed 35042 reproduced
100% success, 0.266 m p95 drift, and 0.191 m at turn completion. This validates
the supported-turn intervention but still misses the 0.15 m final gate.

Continue the same reward unchanged for another 250 updates to measure whether
the compactness learning curve is still improving before changing the weight.
Job `6a9219f4984507d9db4eaa9c` explicitly loaded `model_1747.pt` and resumed at
iteration 1747/1997 with the same source configuration and no NaN terminations.

Checkpoint 1996 retained 100% strict success in every official 256-episode
bucket and improved standing p95 drift to 0.245 m, with 0.173 m at turn
completion. Seed 35042 reproduced 100% success, 0.246 m final p95, and 0.175 m
at turn completion. The unchanged learning curve is still improving but more
slowly than the first supported-penalty stage.

## Experiment 6: stronger supported and recovery compactness

Warm-start checkpoint 1996 and double the supported-spin and post-turn planar
speed multipliers from 4x to 8x. Keep the entry multiplier at 1x and preserve
every other reward, reset, observation, termination, and task-gate setting.
This is the next controlled dose on the replicated compactness intervention.
Promote only if 100% success is preserved and p95 standing drift improves.

HF job `6a921ecd45686a1580c132b6` uses source commit `ed09185`. It explicitly
loaded `model_1996.pt`, resumed at iteration 1996/2246 with 2,048 environments,
and showed finite rewards and no NaN terminations during initialization.

Checkpoint 2245 retained 100% strict stable success in all five official
256-episode buckets. Standing p95 maximum drift improved from 0.245 m to
0.230 m, with 0.166 m accumulated by turn completion and 0.224 m at the final
stand. The improvement is real but small, and partial headstand starts regressed
to 0.372--0.548 m p95 maximum drift. This shows diminishing returns from merely
increasing a squared-velocity proxy; reward-v7 remains behind the 0.15 m final
gate and is not yet the final candidate.

## Experiment 7: direct displacement alignment

Warm-start checkpoint 2245 and preserve the proven entry, supported 180-degree
turn, strict 0.4-second stable-stand success, and 30-point terminal success
reward. Add two terms that optimize the measured failure directly:

- a dense -20 weighted squared displacement from the episode's reset position;
- a one-shot +15 compact-success bonus, paid only on strict success and graded
  by `exp(-(maximum_drift / 0.15 m)^2)`.

The original terminal bonus is unchanged, so a compactness failure cannot erase
the task objective. The new bonus cannot be farmed and maximum drift includes
the whole trajectory. This experiment tests direct objective alignment after
the velocity-proxy dose response flattened. Promote only if standing success
remains 100% and p95 maximum drift improves materially; checkpoint 2245 and the
earlier checkpoint 1249 remain immutable fallbacks.

The 64-environment smoke loaded checkpoint 2245 and exercised both new terms
with finite values and zero NaN terminations. It was canceled immediately after
verification. The full 2,048-environment job then trained 250 updates from the
same clean fallback and completed normally at checkpoint 2494.

Checkpoint 2494 again achieved 100% strict success in all five official
256-episode buckets. Standing p95 maximum drift improved only from 0.230 m to
0.222 m; turn-completion drift improved from 0.166 m to 0.153 m and final drift
from 0.224 m to 0.220 m. This is a smaller gain than needed and rejects further
dose increases of the same whole-root proxy without first fixing the metric.

The maximum-drift metric follows the trunk root through a maneuver that must
rotate the trunk around a ground-planted head, so it mixes necessary kinematic
excursion with undesirable pivot sliding. Before further reward changes,
instrument the `jaw_soft` head body's displacement during valid head-only
support. Compactness should mean a planted head pivot plus a final stand near
the starting position, rather than an immobile trunk throughout inversion.

The corrected evaluation-only diagnostic measured standing-start p95 head-COM
drift of 0.101 m during valid head-only support, versus 0.011 m in recovery-only
episodes with no supported yaw. This confirms real pivot sliding during the
turn, while also showing that the 0.223 m maximum trunk excursion is not itself
the right compactness target. Checkpoint 2494 finished 0.221 m from its initial
root position.

## Experiment 8: phase-correct compactness

Warm-start checkpoint 2494 and preserve all task/success gates. Correct the
compactness semantics without increasing their weights:

- during valid supported yaw, the 8x motion cost reads `jaw_soft` head-COM
  planar velocity rather than trunk-root velocity;
- add a -20 squared head-COM displacement cost relative to first valid support;
- apply the existing -20 root-displacement cost only after the 180-degree turn;
- grade the +15 compact-success event by final root displacement, not maximum
  trunk excursion during the inverted arc.

This separates pivot slip from necessary body kinematics and separately asks
the recovery to finish near its starting point. Promote if 100% official success
is preserved and both standing p95 head drift and final root displacement cross
their 0.10 m / 0.20 m gates.

The first smoke exposed an unresolved `SceneEntityCfg.body_ids` slice before its
first PPO update. The corrected smoke resolves `jaw_soft` from the runtime asset,
loaded checkpoint 2494, produced finite nonzero pivot-displacement rewards, and
reported zero NaN terminations. It was canceled after verification.
