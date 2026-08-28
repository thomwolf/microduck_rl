# Head-spin training

`Mjlab-HeadSpin-Flat-MicroDuck` trains one episodic sequence:

1. leave a stable stand;
2. put the flat top of the head on the terrain with both feet clear;
3. accumulate at least π radians of net world-yaw in the requested direction;
4. recover to the HOME standing pose and settle.

Yaw progress is counted only during verified head-only support. It is signed,
so alternating left/right rocking cannot fake a half-turn, and it is paid as
new max-frontier increments, so holding a pose cannot farm reward. Landing
rewards remain closed until the supported-yaw frontier reaches π.

## Observation contract

The policy keeps the standard 61D actor observation. The existing four-value
head-command slot is repurposed for `[enter, spin, recover, direction]`:

- `[1, 0, 0, +1]`: seek valid head support;
- `[0, 1, 0, +1]`: remain supported and turn;
- `[0, 0, 1, +1]`: the half-turn is complete; recover to the feet.

This state is necessary because absolute yaw is intentionally absent from the
proprioceptive observation: after 180° a symmetric headstand otherwise looks
the same to a feed-forward policy. Deployment must reproduce this small state
machine from IMU yaw integration plus a head-support estimate.

## Validate before a long run

Run the focused CPU tests:

```bash
uv run pytest tests/test_head_spin.py -q
```

Then submit the required 64-environment, five-iteration smoke test:

```bash
uv run python -m mjlab_microduck.train_cli Mjlab-HeadSpin-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max_iterations 5 \
  --hf-jobs --namespace pollen-robotics --run-name headspin-smoke \
  --flavor l4x1 --timeout 1h
```

Only after the smoke test has finite observations/rewards and writes a
checkpoint, start the curriculum-spanning run:

```bash
uv run python -m mjlab_microduck.train_cli Mjlab-HeadSpin-Flat-MicroDuck \
  --env.scene.num-envs 4096 --agent.max_iterations 8000 \
  --hf-jobs --namespace pollen-robotics --run-name headspin-v1 \
  --flavor l4x1 --timeout 12h
```

The reset curriculum initially samples standing entry, incomplete headstand,
and already-complete recovery states in a 25/55/20 mix. It shifts toward
60/30/10 by iteration 6000. This is deliberate: spinning and exiting are
learned before most rollouts demand the full standing-to-standing sequence.

## What to inspect in video and metrics

- head support is on the flat top, not the beak, face, or side shell;
- both feet are visibly clear while yaw progress grows;
- accumulated supported yaw reaches at least 3.14 rad;
- planar drift stays small and the spin does not become ballistic;
- the final trunk height is near 0.115 m, tilt settles, and the robot remains
  standing rather than merely touching its feet down.

The W&B metrics deliberately separate `head_spin_turn_complete` (the turn was
earned rather than pre-seeded), `head_spin_full_success` (standing-start full
sequence), and `head_spin_recovery_success` (recovery-curriculum bucket).

If the policy spins but does not recover, keep the recovery reset bucket high
longer. If it recovers from reverse-curriculum starts but never finds the entry
from standing, delay the shift toward standing starts rather than increasing
always-on upright or smoothness penalties; those terms directly oppose the
maneuver during discovery.
