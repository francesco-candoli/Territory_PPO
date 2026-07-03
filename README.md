# Territory PPO

A grid "territory conquest" environment where two agents race to claim empty
cells. This project trains a single PPO agent (agent id 1) against a fixed
rule-based `StaticAgent` (agent id 2), and provides tooling to play matches,
evaluate win-rates, and run a round-robin tournament between reward variants.

This is a restructured, runnable version of the original `AAS_project (10).ipynb`
notebook: the notebook remains the source of truth for the algorithms, and this
project only modularizes the code into importable files and adds a CLI. No
algorithm or reward logic was changed, other than making the PPO network read
its grid size from the observation space instead of a notebook-global `N`.

## File layout

```
Config.py        hyperparameters, reward registry, default paths, naming helpers
Environment.py   TerritoryEnv (gymnasium.Env) and the Moves action enum
Rewards.py       RewardFunction ABC + DenseReward, TerminalReward, HybridReward
Buffer.py        rollout storage used by PPOAgent
StaticAgent.py   rule-based opponent (prefers adjacent empty cells)
PPOAgent.py      PolicyNetwork (CNN) + PPOAgent (act/update/save/load)
Trainer.py       Trainer (rollout + PPO update loop), training-log save/load,
                 plot_training
Evaluator.py     play_match, evaluate_vs_static, evaluate_ppo_vs_ppo, GIF
                 makers, tournament (run_series, run_tournament,
                 summarize/save/load/plot_tournament)
utils.py         device auto-detect, seeding, best-effort GIF display
main.py          argparse CLI: smoke / train / play / eval / tournament

data/models/     drop trained .pt weights here (created empty)
data/history/    drop training-history .json files here (created empty)
data/gif/        default output folder for match GIFs
```

## Install

```
pip install -r requirements.txt
```

Requires Python 3.9+. A CUDA GPU is used automatically if available
(`torch.cuda.is_available()`), otherwise everything runs on CPU.

## Commands

All configuration lives in `Config.py`; any hyperparameter not passed on the
command line falls back to its value there. See `instructions.txt` for
concrete copy-paste examples of every command.

- **`smoke`** — builds a `TerritoryEnv`, steps it a few times with two
  `StaticAgent`s, and asserts the observation shape `(4, N, N)` and the action
  space. No weights required; use it to confirm the environment runs.

- **`train`** — trains one PPO agent (agent id 1) against the fixed
  `StaticAgent` (agent id 2) using one of the three reward variants
  (`Dense`, `Terminal`, `Hybrid`). Saves weights to `data/models/` and the
  training log to `data/history/` using the naming convention
  `weight_N<N>_<Variant>_<steps>.pt` / `history_N<N>_<Variant>_<steps>.json`
  (override with `--name`). `Terminal` defaults to `gamma=0.999`; `Dense` and
  `Hybrid` default to `gamma=0.99` — gamma only affects training (it shapes
  returns/advantages during `update()`), it has no effect on evaluating
  frozen weights.

- **`play`** — plays a single match (`static_vs_static`, `ppo_vs_static`, or
  `ppo_vs_ppo`) and always saves a GIF to `data/gif/`, displaying it inline if
  run in a notebook/IPython session. Prints the final cell counts and winner.

- **`eval`** — plays many matches (default 100) of `ppo_vs_static` or
  `ppo_vs_ppo` and reports the win-rate and mean cells. Evaluation samples
  stochastically from the policy by default (pass `--deterministic` for
  argmax actions).

- **`tournament`** — round-robins all four competitors (Dense, Terminal,
  Hybrid PPO variants, plus the Static baseline) across a set of seeds
  (default `0 44 999`), prints the cross-play win-rate table and ranking, and
  saves the raw results to `data/tournament_results.json`. It does not
  generate the heatmap plot (available separately as `Evaluator.plot_tournament`
  if you want it).

## Fixed setup

The project always trains/evaluates a single PPO learner (agent id 1) against
a fixed `StaticAgent` opponent (agent id 2) — not independent multi-agent PPO.
`N_AGENT` is fixed at 2 throughout; `MAX_STEP` defaults to `N**2 * 3` for
whatever `N` is in effect unless `--max-step` is passed explicitly.
