# dino_master

Chrome Dino NEAT agent trained in a **pure-Python simulator**, then verified on the real game.

## Result

The shipped champion (`models/champion-7553.pkl`) cleared **40,000 steps** on the public
[wayou/t-rex-runner](https://wayou.github.io/t-rex-runner/) game:

| run | steps | time at 60 Hz |
|---|---:|---:|
| remote 1 | 40,000 | ~10.7 min |
| remote 2 | 40,000 | ~10.7 min |
| remote 3 | 40,000 | ~10.7 min |
| local vendored | 40,000 | ~10.7 min |
| this repo (`dino-eval`) | 40,000 | ~10.7 min |

The genome is tiny: 3 nodes / 4 connections.

## Quick start

```bash
uv sync
uv run dino-train --generations 50 --population 500 --max-steps 30000 \
  --episodes-per-gen 4 --best models/best.pkl

# real-game evaluation (Google Chrome)
uv sync --extra real
uv run dino-eval --genome models/champion-7553.pkl --episodes 3 \
  --max-steps 40000 --out results/eval.json

# watch the champion play
uv run dino-spectate --genome models/champion-7553.pkl
```

Set `DINO_CHROME_EXECUTABLE` if Chrome is not auto-detected. `DINO_HEADLESS=1` runs the
real-game checks headless.

## How it works

- `src/dino_master/sim.py` — exact port of Dino physics, obstacles, gaps and collision
  boxes from the original `index.js`.
- `src/dino_master/features.py` — 24 observation features, action mask, duck hold,
  jump cooldown, curriculum and fitness.
- `src/dino_master/train.py` — NEAT evolution inside the simulator. 500 genomes, 60 Hz,
  no browser. Fitness: `steps + 5 * clears + 1000` for reaching `max_steps`.
- `src/dino_master/real_game.py` — Playwright/Chrome evaluation on the public game.
- `src/dino_master/spectate.py` — visible window with the current genome.

Training uses 4 independent episodes per generation and curriculum speeds
`6 → 8 → 10 → 13` (5 generations each), then natural acceleration.

## Repo layout

```text
src/dino_master/neat.ini  NEAT config (24 → 3, pop 500)
models/                champion pickle
results/               reproducibility evidence
src/dino_master/       simulator, features, training, eval, spectator
tests/                 browser-free unit tests
```

## Limitations

- The champion is a tiny reactive network; it is validated on the public game, but a
  single report is not an exhaustive benchmark.
- Real-game evaluation needs Google Chrome and network access.
- `step_delay=0.016` matches 60 Hz; the browser window must stay focused/active.
- The failed pterodactyl-bonus experiment is intentionally not part of this repository.

## Credits

Chrome Dino by the Chromium Authors; public runner by
[wayou](https://github.com/wayou/t-rex-runner). This repository contains no vendored
game assets.
