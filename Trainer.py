import json
import numpy as np
import matplotlib.pyplot as plt
from Environment import Moves

class Trainer:
  def __init__(self, env, agents, reward_fns, rollout_len=2048, device="cpu"):
    self.env = env
    self.agents = agents
    self.reward_fns = reward_fns
    self.rollout_len = rollout_len
    self.device = device
    # gli agenti che imparano = quelli che hanno una reward function
    self.learners = [aid for aid in agents if aid in reward_fns]

  def _is_ppo(self, aid):
    return aid in self.reward_fns

  def train(self, total_steps=100_000, log_every=5):
    obs, info = self.env.reset()
    prev_info = info
    step_count = 0
    episode = 0
    ep_return = {aid: 0.0 for aid in self.learners}
    history = []
    update_stats = []

    while step_count < total_steps:
      # --- raccogli un'azione per ogni agente ---
      actions = []
      pending = {}   # aid -> (obs, idx, logp, val) per gli agenti PPO

      #CHECCO: 1. calcola action, logits(actor), action-state value(critic)

      for aid, agent in self.agents.items():
        idx, logp, val = agent.act(obs[aid])
        actions.append((aid, list(Moves)[idx]))
        if self._is_ppo(aid):
          pending[aid] = (obs[aid], idx, logp, val)

      #CHECCO: 2. l'ambiente viene portato in avanti

      # --- avanza l'ambiente ---
      next_obs, terminated, truncated, info = self.env.step(actions)
      done = terminated or truncated

      #CHECCO: 3. sul nuovo state calcola la reward, poi immagazzina i valori dello stato precedente:
      #           observation, idx_agente, logits(vecchi), value(vecchio),
      #           reward acquisita, e se finisce l'episodio

      # --- calcola la reward e immagazzina, solo per i PPO ---
      for aid in self.learners:
        o, idx, logp, val = pending[aid]
        r = self.reward_fns[aid].compute(prev_info, info, aid, terminated, truncated)
        self.agents[aid].buffer.store(o, idx, logp, val, r, done)
        ep_return[aid] += r

      obs = next_obs
      prev_info = info
      step_count += 1

      # --- fine episodio: reset ---
      if done:
        episode += 1
        if episode % log_every == 0:
          cells = info["cells"]
          winner = max(cells, key=cells.get)
          avg_ret = {aid: round(ep_return[aid], 4) for aid in self.learners}
          history.append({"episode": episode, "cells": dict(cells),
                          "winner": winner, "returns": avg_ret})
          if episode % 10 == 0:
            print(f"ep {episode:4d} | step {step_count:6d} | "
                  f"cells {dict(cells)} | winner {winner} | ret {avg_ret}")
        obs, info = self.env.reset()
        prev_info = info
        ep_return = {aid: 0.0 for aid in self.learners}

      #CHECCO: 4. continua a immagazzinare finchè il rollout non è pieno, fa l'update che dentro di sè chiamerà GAE

      # --- update PPO ogni rollout_len step ---
      if step_count % self.rollout_len == 0:
        for aid in self.learners:
          # bootstrap value dello stato corrente (rollout troncato)
          if done:
            last_value = 0.0
          else:
            _, _, last_value = self.agents[aid].act(obs[aid])
          stats = self.agents[aid].update(last_value=last_value)   # <-- cattura
          if aid == self.learners[0]:        # logga le stats del primo learner
            update_stats.append(stats)


    return history, update_stats


def save_training_log(history, update_stats, path, meta=None):
    data = {"meta": meta or {}, "history": history, "update_stats": update_stats}
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Log salvato in {path} "
          f"({len(history)} episodi loggati, {len(update_stats)} update)")

def load_training_log(path):
    with open(path) as f:
        data = json.load(f)
    for h in data["history"]:
        h["cells"]   = {int(k): v for k, v in h["cells"].items()}
        h["returns"] = {int(k): v for k, v in h["returns"].items()}
    return data["meta"], data["history"], data["update_stats"]


def plot_training(history, update_stats, learner_id):

    window = 20

    # Win rate
    wins = [1 if h["winner"] == learner_id else 0 for h in history]
    winrate = [np.mean(wins[max(0, i-window):i+1]) for i in range(len(wins))]

    plt.figure(figsize=(14, 4))
    step = 2
    x = np.arange(len(winrate))[::step]

    plt.plot(x, winrate[::step])
    plt.title(f"Win-rate (moving avg {window}) agent {learner_id}")
    plt.ylim(0, 1)
    plt.axhline(0.5, ls="--", c="gray")
    plt.xlabel("Episode")
    plt.ylabel("Win rate")
    plt.grid(True)

    # Final cells
    cells = [h["cells"][learner_id] for h in history]

    plt.figure(figsize=(14, 4))
    plt.plot(x, cells[::step])
    plt.title("Cells owned at the end of each episode")
    plt.xlabel("Episode")
    plt.ylabel("Cells")
    plt.grid(True)

    # Policy entropy
    entropy = [s["entropy"] for s in update_stats]
    x = np.arange(len(entropy))[::step]

    plt.figure(figsize=(14, 4))
    plt.plot(x, entropy[::step])
    plt.title("Policy entropy")
    plt.xlabel("Update")
    plt.ylabel("Entropy")
    plt.grid(True)

    # Value loss
    value_loss = [s["value_loss"] for s in update_stats]

    plt.figure(figsize=(14, 4))
    plt.plot(x, value_loss[::step])
    plt.title("Value loss")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.grid(True)
