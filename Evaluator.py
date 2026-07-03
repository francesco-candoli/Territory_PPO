import json
import random
from itertools import combinations
import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt
import Config
from Environment import Moves, TerritoryEnv
from PPOAgent import PPOAgent
from StaticAgent import StaticAgent

def play_match(env, agents, seed=None, render=False, gif_path="match.gif", fps=10):
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    obs, info = env.reset()
    frames = [env.render()] if render else []

    terminated = truncated = False
    while not (terminated or truncated):
        # CHECCO REMINDER azioni nel formato (agent_id, Move)
        actions = []
        for aid, agent in agents.items():
            idx, _, _ = agent.act(obs[aid])
            move = list(Moves)[idx]
            actions.append((aid, move))

        obs, terminated, truncated, info = env.step(actions)
        if render:
            frames.append(env.render())

    if render:
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"GIF salvata in {gif_path}")

    return info["cells"]

def run_greedy_match(N=10, n_agent=4, max_step=400, seed=0):
    env = TerritoryEnv(N=N, n_agent=n_agent, max_step=max_step)
    agents = {aid: StaticAgent(agent_id=aid) for aid in env.agent_ids}
    cells = play_match(env, agents, seed=seed, render=True, gif_path="greedy_match.gif")
    winner = max(cells, key=cells.get)
    print("Celle finali:", cells)
    print("Vincitore:", winner)
    return cells

def evaluate_vs_static(ppo_agent, N=Config.N, n_agent=Config.N_AGENT, n_games=Config.DEFAULT_GAMES, max_step=Config.MAX_STEP, deterministic=False):
    ppo_agent.actor.eval(); ppo_agent.critic.eval()
    wins, my_cells, opp_cells = 0, [], []

    for g in range(n_games):
        env = TerritoryEnv(N=N, n_agent=n_agent, max_step=max_step)
        agents = {ppo_agent.agent_id: ppo_agent}
        for aid in env.agent_ids:
            if aid != ppo_agent.agent_id:
                agents[aid] = StaticAgent(agent_id=aid)

        obs, info = env.reset()
        done = False
        while not done:
            actions = []
            for aid, ag in agents.items():
                idx, _, _ = ag.act(obs[aid], deterministic=deterministic)
                actions.append((aid, list(Moves)[idx]))
            obs, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

        cells = info["cells"]
        mine = cells[ppo_agent.agent_id]
        opp = max(v for k, v in cells.items() if k != ppo_agent.agent_id)
        my_cells.append(mine); opp_cells.append(opp)
        if mine > opp: wins += 1
    print(f"Win-rate su {n_games} partite: {wins/n_games:.2%}")
    print(f"Celle medie  -> PPO: {np.mean(my_cells):.1f} | static: {np.mean(opp_cells):.1f}")
    return wins / n_games

def make_ppo_vs_static_gif(ppo_agent, N, n_agent=2, max_step=300,
                           seed=0, gif_path="ppo_vs_static.gif",
                           fps=10, deterministic=False):

    ppo_agent.actor.eval(); ppo_agent.critic.eval()

    env = TerritoryEnv(N=N, n_agent=n_agent, max_step=max_step)
    agents = {ppo_agent.agent_id: ppo_agent}
    for aid in env.agent_ids:
        if aid != ppo_agent.agent_id:
            agents[aid] = StaticAgent(agent_id=aid)

    cells = play_match(env, agents, seed=seed, render=True,
                       gif_path=gif_path, fps=fps)

    winner = max(cells, key=cells.get)
    label = "PPO" if winner == ppo_agent.agent_id else f"static({winner})"
    print(f"Celle finali: {cells}  ->  vince {label}")
    return cells

def evaluate_ppo_vs_ppo(ppo_a, ppo_b, N=Config.N, n_games=Config.DEFAULT_GAMES, max_step=Config.MAX_STEP, deterministic=False):
    
    for ag in (ppo_a, ppo_b):
        ag.actor.eval(); ag.critic.eval()

    wins_a = wins_b = ties = 0
    cells_a, cells_b = [], []

    for g in range(n_games):
        slot = {1: ppo_a, 2: ppo_b}
        env = TerritoryEnv(N=N, n_agent=2, max_step=max_step)
        obs, info = env.reset()
        done = False
        while not done:
            actions = []
            for aid, agent in slot.items():
                idx, _, _ = agent.act(obs[aid], deterministic=deterministic)
                actions.append((aid, list(Moves)[idx]))
            obs, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

        cells = info["cells"]
        ca = cells[1]
        cb = cells[2]
        cells_a.append(ca); cells_b.append(cb)
        if   ca > cb: wins_a += 1
        elif cb > ca: wins_b += 1
        else:         ties  += 1

    print(f"Su {n_games} partite -> A vince {wins_a/n_games:.0%} | "
          f"B vince {wins_b/n_games:.0%} | pareggi {ties/n_games:.0%}")
    print(f"Celle medie -> A: {np.mean(cells_a):.1f} | B: {np.mean(cells_b):.1f}")
    return wins_a/n_games, wins_b/n_games, ties/n_games

def make_ppo_vs_ppo_gif(ppo_a, ppo_b, N, max_step=Config.MAX_STEP, seed=0, gif_path="ppo_vs_ppo.gif", fps=10, deterministic=False):
    for ag in (ppo_a, ppo_b):
        ag.actor.eval(); ag.critic.eval()
    if seed is not None:
        random.seed(seed); np.random.seed(seed)

    env = TerritoryEnv(N=N, n_agent=2, max_step=max_step)
    slot = {1: ppo_a, 2: ppo_b}

    obs, info = env.reset()
    frames = [env.render()]
    terminated = truncated = False
    while not (terminated or truncated):
        actions = []
        for aid, agent in slot.items():
            idx, _, _ = agent.act(obs[aid], deterministic=deterministic)
            actions.append((aid, list(Moves)[idx]))
        obs, terminated, truncated, info = env.step(actions)
        frames.append(env.render())

    imageio.mimsave(gif_path, frames, fps=fps)
    cells = info["cells"]
    label = ("vince A (agente 1)" if cells[1] > cells[2] else "vince B (agente 2)" if cells[2] > cells[1] else "pareggio")
    print(f"Final cells: {cells}  ->  {label} | GIF: {gif_path}")
    return cells


def load_frozen_ppo(path, agent_id, obs_space, act_space, device="cpu"):
    ag = PPOAgent(obs_space, act_space, agent_id=agent_id, device=device)
    ag.load(path)
    ag.actor.eval()
    ag.critic.eval()
    return ag


def build_competitors(model_paths, obs_space, act_space, device="cpu"):
    competitors = {
        name: (lambda aid, p=path: load_frozen_ppo(p, aid, obs_space, act_space, device))
        for name, path in model_paths.items()
    }
    competitors["Static"] = lambda aid: StaticAgent(agent_id=aid)
    return competitors


def run_series(agent1, agent2, N=Config.N, n_games=Config.DEFAULT_GAMES, max_step=Config.MAX_STEP):
    env_match = TerritoryEnv(N=N, n_agent=2, max_step=max_step)
    agents = {agent1.agent_id: agent1, agent2.agent_id: agent2}

    w1 = w2 = t = 0
    for _ in range(n_games):
        cells = play_match(env_match, agents) 
        if   cells[1] > cells[2]: w1 += 1
        elif cells[2] > cells[1]: w2 += 1
        else:                     t  += 1
    return w1 / n_games, w2 / n_games, t / n_games


def run_tournament(competitors, N=Config.N, n_games=Config.DEFAULT_GAMES, max_step=Config.MAX_STEP, seeds=Config.TOURNAMENT_SEEDS):
    names = list(competitors.keys())
    results = {a: {b: [] for b in names if b != a} for a in names}
    ties = {}

    for a, b in combinations(names, 2):
        agent1 = competitors[a](1)   # a gioca come id 1
        agent2 = competitors[b](2)   # b gioca come id 2

        for seed in seeds:
            # riproducibilita': env (random, np), StaticAgent (random)
            # e sampling della policy PPO (torch)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            wa, wb, t = run_series(agent1, agent2, N=N,
                                   n_games=n_games, max_step=max_step)
            print(f"=== {a} (id 1) vs {b} (id 2) | seed {seed} -> "
                  f"{a} {wa:.0%} | {b} {wb:.0%} | pareggi {t:.0%}")
            results[a][b].append(wa)
            results[b][a].append(wb)
            ties.setdefault((a, b), []).append(t)

    return results, ties

def summarize_tournament(results, ties):
    names = list(results.keys())
    col_w = max(len(n) for n in names) + 2

    # matrice: riga = concorrente, colonna = avversario, media +- std sui seed
    print(" " * col_w + "".join(f"{b:>{col_w + 10}}" for b in names))
    for a in names:
        row = f"{a:<{col_w}}"
        for b in names:
            if a == b:
                row += f"{'-':>{col_w + 10}}"
            else:
                m, s = np.mean(results[a][b]), np.std(results[a][b])
                row += f"{f'{m:.0%} +- {s:.0%}':>{col_w + 10}}"
        print(row)

    print("\nPareggi per coppia:")
    for (a, b), t in ties.items():
        print(f"  {a} vs {b}: {np.mean(t):.0%} +- {np.std(t):.0%}")

    # classifica: media dei win-rate medi contro ciascun avversario
    print("\nClassifica (win-rate medio cross-play):")
    scores = {a: float(np.mean([np.mean(v) for v in results[a].values()]))
              for a in names}
    for rank, (a, s) in enumerate(sorted(scores.items(), key=lambda kv: -kv[1]), 1):
        print(f"  {rank}. {a}: {s:.1%}")
    return scores

def save_tournament(results, ties, path="tournament_results.json"):
    payload = {
        "results": results,  # liste di float: gia' serializzabili
        "ties": {f"{a}__vs__{b}": v for (a, b), v in ties.items()},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Risultati salvati in {path}")

def load_tournament(path="tournament_results.json"):
    with open(path) as f:
        payload = json.load(f)
    ties = {tuple(k.split("__vs__")): v for k, v in payload["ties"].items()}
    return payload["results"], ties

def plot_tournament(results, title="Cross-play: win-rate di riga contro colonna"):
    names = list(results.keys())
    n = len(names)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if a != b:
                M[i, j] = np.mean(results[a][b])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(n), labels=names)
    ax.set_yticks(range(n), labels=names)
    ax.set_xlabel("Avversario")
    ax.set_ylabel("Concorrente")
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="win-rate")
    plt.tight_layout()