import argparse
import os

import Config
import utils
from Environment import Moves, TerritoryEnv
from StaticAgent import StaticAgent
from PPOAgent import PPOAgent
from Trainer import Trainer, save_training_log
from Evaluator import (
    build_competitors,
    evaluate_ppo_vs_ppo,
    evaluate_vs_static,
    load_frozen_ppo,
    play_match,
    run_tournament,
    save_tournament,
    summarize_tournament,
)


def get_max_step(n, max_step):
    return max_step if max_step is not None else Config.default_max_step(n)


def cmd_smoke(args):
    max_step = get_max_step(args.N, args.max_step)
    env = TerritoryEnv(N=args.N, n_agent=Config.N_AGENT, max_step=max_step)
    agents = {1: StaticAgent(agent_id=1), 2: StaticAgent(agent_id=2)}

    obs, info = env.reset()
    for aid in env.agent_ids:
        assert obs[aid].shape == (4, args.N, args.N), f"unexpected observation shape for agent {aid}: {obs[aid].shape}"
    assert env.action_space.n == len(Moves), "the action  space does not match Moves"

    terminated = truncated = False
    for _ in range(args.steps):
        if terminated or truncated:
            break
        actions = []
        for aid, agent in agents.items():
            idx, _, _ = agent.act(obs[aid])
            actions.append((aid, list(Moves)[idx]))
        obs, terminated, truncated, info = env.step(actions)

    print(f"the observation shape is ok: {obs[1].shape} | the action space is ok: Discrete({env.action_space.n})")
    print(f"final cells after {info['step_done']} steps: {info['cells']}")


def cmd_train(args):
    max_step = get_max_step(args.N, args.max_step)
    gamma = args.gamma if args.gamma is not None else Config.GAMMA_BY_VARIANT[args.reward]
    utils.set_seed(args.seed)
    device = utils.get_device()

    env = TerritoryEnv(N=args.N, n_agent=Config.N_AGENT, max_step=max_step)
    ppo = PPOAgent(env.observation_space, env.action_space, agent_id=1,
                   lr=args.lr, gamma=gamma, lam=Config.LAM, clip=Config.CLIP,
                   value_coef=Config.VALUE_COEF, entropy_coef=Config.ENTROPY_COEF,
                   epochs=Config.EPOCHS, minibatch_size=Config.MINIBATCH_SIZE,
                   max_grad_norm=Config.MAX_GRAD_NORM, device=device)
    static = StaticAgent(agent_id=2)
    reward_cls = Config.REWARD_REGISTRY[args.reward]
    reward_fn = reward_cls(win_bonus=Config.WIN_BONUS) if args.reward == "Hybrid" else reward_cls()

    agents = {1: ppo, 2: static}
    reward_fns = {1: reward_fn}
    trainer = Trainer(env, agents, reward_fns, rollout_len=args.rollout_len, device=device)

    print(f"Training {args.reward} PPO | N={args.N} max_step={max_step} steps={args.steps} gamma={gamma} lr={args.lr} rollout_len={args.rollout_len} device={device}")
    history, update_stats = trainer.train(total_steps=args.steps)

    name = args.name if args.name is not None else f"N{args.N}_{args.reward}_{args.steps}"
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(Config.HISTORY_DIR, exist_ok=True)
    weight_path = os.path.join(Config.MODELS_DIR, f"weight_{name}.pt")
    history_path = os.path.join(Config.HISTORY_DIR, f"history_{name}.json")

    ppo.save(weight_path)
    meta = {"N": args.N, "reward": args.reward, "steps": args.steps, "max_step": max_step, "gamma": gamma, "lr": args.lr, "rollout_len": args.rollout_len, "seed": args.seed}
    save_training_log(history, update_stats, history_path, meta=meta)
    print(f"Weights saved to {weight_path}")


def cmd_play(args):
    max_step = get_max_step(args.N, args.max_step)
    utils.set_seed(args.seed)
    device = utils.get_device()

    env = TerritoryEnv(N=args.N, n_agent=Config.N_AGENT, max_step=max_step)

    if args.mode == "static_vs_static":
        agents = {1: StaticAgent(agent_id=1), 2: StaticAgent(agent_id=2)}

    elif args.mode == "ppo_vs_static":
        if not args.weights_a:
            raise SystemExit("play --mode ppo_vs_static requires --weights-a")
        
        ppo = load_frozen_ppo(args.weights_a, 1, env.observation_space, env.action_space, device)
        agents = {1: ppo, 2: StaticAgent(agent_id=2)}
    else:  
        if not args.weights_a or not args.weights_b:
            raise SystemExit("play --mode ppo_vs_ppo requires --weights-a and --weights-b")
        
        ppo_a = load_frozen_ppo(args.weights_a, 1, env.observation_space, env.action_space, device)
        ppo_b = load_frozen_ppo(args.weights_b, 2, env.observation_space, env.action_space, device)
        agents = {1: ppo_a, 2: ppo_b}

    os.makedirs(Config.GIF_DIR, exist_ok=True)
    suffix = f"_seed{args.seed}" if args.seed is not None else ""

    gif_path = os.path.join(Config.GIF_DIR, f"{args.mode}{suffix}.gif")

    cells = play_match(env, agents, seed=args.seed, render=True, gif_path=gif_path, fps=args.fps)
    winner = max(cells, key=cells.get)

    print(f"Final cells: {cells}")
    print(f"Winner: agent {winner}")

def cmd_eval(args):
    max_step = get_max_step(args.N, args.max_step)
    utils.set_seed(args.seed)
    device = utils.get_device()

    env = TerritoryEnv(N=args.N, n_agent=Config.N_AGENT, max_step=max_step)

    if args.mode == "ppo_vs_static":

        if not args.weights:
            raise SystemExit("eval --mode ppo_vs_static requires --weights")
        ppo = load_frozen_ppo(args.weights, 1, env.observation_space, env.action_space, device)
        evaluate_vs_static(ppo, N=args.N, n_agent=Config.N_AGENT, n_games=args.games,
                            max_step=max_step, deterministic=args.deterministic)
        
    else:  
        if not args.weights_a or not args.weights_b:
            raise SystemExit("eval --mode ppo_vs_ppo requires --weights-a and --weights-b")
        ppo_a = load_frozen_ppo(args.weights_a, 1, env.observation_space, env.action_space, device)
        ppo_b = load_frozen_ppo(args.weights_b, 2, env.observation_space, env.action_space, device)

        evaluate_ppo_vs_ppo(ppo_a, ppo_b, N=args.N, n_games=args.games,max_step=max_step, deterministic=args.deterministic)


def cmd_tournament(args):
    max_step = get_max_step(args.N, args.max_step)
    device = utils.get_device()

    env = TerritoryEnv(N=args.N, n_agent=Config.N_AGENT, max_step=max_step)
    model_paths = {
        "Dense": args.dense_weights,
        "Terminal": args.terminal_weights,
        "Hybrid": args.hybrid_weights,
    }
    competitors = build_competitors(model_paths, env.observation_space, env.action_space, device)

    results, ties = run_tournament(competitors, N=args.N, n_games=args.games, max_step=max_step, seeds=tuple(args.seeds))

    summarize_tournament(results, ties)

    os.makedirs(Config.DATA_DIR, exist_ok=True)
    save_tournament(results, ties, path=Config.TOURNAMENT_RESULTS_PATH)


def build_parser():
    parser = argparse.ArgumentParser(prog="main.py", description="territoryEnv PPO project CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_smoke = sub.add_parser("smoke", help="smoke test the environment")
    p_smoke.add_argument("--N", type=int, default=Config.N)
    p_smoke.add_argument("--max-step", type=int, default=None, help="default: N*N*3")
    p_smoke.add_argument("--steps", type=int, default=20)
    p_smoke.set_defaults(func=cmd_smoke)

    p_train = sub.add_parser("train", help="train a PPO agent vs the fixed StaticAgent")
    p_train.add_argument("--reward", required=True, choices=list(Config.REWARD_REGISTRY.keys()))
    p_train.add_argument("--N", type=int, default=Config.N)
    p_train.add_argument("--steps", type=int, default=Config.DEFAULT_TRAIN_STEPS, help="total env steps")
    p_train.add_argument("--max-step", type=int, default=None, help="default: N*N*3")
    p_train.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    p_train.add_argument("--gamma", type=float, default=None)
    p_train.add_argument("--rollout-len", type=int, default=Config.ROLLOUT_LEN)
    p_train.add_argument("--seed", type=int, default=None)
    p_train.add_argument("--name", type=str, default=None, help="override the default naming")
    p_train.set_defaults(func=cmd_train)

    p_play = sub.add_parser("play", help="Play a single match and save a GIF")
    p_play.add_argument("--mode", required=True, choices=["static_vs_static", "ppo_vs_static", "ppo_vs_ppo"])
    p_play.add_argument("--weights-a", type=str, default=None, help="PPO weights for ppo_vs_static, or first PPO for ppo_vs_ppo")
    p_play.add_argument("--weights-b", type=str, default=None, help="second PPO weights for ppo_vs_ppo")
    p_play.add_argument("--N", type=int, default=Config.N)
    p_play.add_argument("--max-step", type=int, default=None, help="default: N*N*3")
    p_play.add_argument("--seed", type=int, default=None)
    p_play.add_argument("--fps", type=int, default=Config.DEFAULT_FPS)
    p_play.set_defaults(func=cmd_play)

    p_eval = sub.add_parser("eval", help="run many matches and report win-rate")
    p_eval.add_argument("--mode", required=True, choices=["ppo_vs_static", "ppo_vs_ppo"])
    p_eval.add_argument("--weights", type=str, default=None, help="PPO weights")
    p_eval.add_argument("--weights-a", type=str, default=None, help="first PPO weights")
    p_eval.add_argument("--weights-b", type=str, default=None, help="second PPO weights")
    p_eval.add_argument("--games", type=int, default=Config.DEFAULT_GAMES)
    p_eval.add_argument("--N", type=int, default=Config.N)
    p_eval.add_argument("--max-step", type=int, default=None, help="default: N*N*3")
    p_eval.add_argument("--seed", type=int, default=None)
    p_eval.add_argument("--deterministic", action="store_true", default=False, help="argmax actions")
    p_eval.set_defaults(func=cmd_eval)

    p_tour = sub.add_parser("tournament")
    p_tour.add_argument("--games", type=int, default=Config.DEFAULT_GAMES)
    p_tour.add_argument("--seeds", type=int, nargs="+", default=list(Config.TOURNAMENT_SEEDS))
    p_tour.add_argument("--N", type=int, default=Config.N)
    p_tour.add_argument("--max-step", type=int, default=None, help="default: N*N*3")
    p_tour.add_argument("--dense-weights", type=str, default=Config.DEFAULT_MODEL_PATHS["Dense"])
    p_tour.add_argument("--terminal-weights", type=str, default=Config.DEFAULT_MODEL_PATHS["Terminal"])
    p_tour.add_argument("--hybrid-weights", type=str, default=Config.DEFAULT_MODEL_PATHS["Hybrid"])
    p_tour.set_defaults(func=cmd_tournament)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
