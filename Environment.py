import random
from enum import Enum
from random import shuffle
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class Moves(Enum):
  UP = [0,-1]
  DOWN = [0,1]
  LEFT = [-1,0]
  RIGHT = [1,0]

class TerritoryEnv(gym.Env):

  def __init__(self, N, n_agent, max_step = 1000):
    assert N > 2, "Map size must be greater than 2"
    assert n_agent > 1, "At least 2 agents"
    assert n_agent < 5, "At most 4 agents"

    self.N = N
    self.n_agent = n_agent
    self.max_step = max_step
    self.agent_ids = [i+1 for i in range(n_agent)]

    self.observation_space = spaces.Box(low=0, high=1, shape=(4, self.N, self.N), dtype=np.float32)
    self.action_space = spaces.Discrete(len(Moves))

    self.reset()

  def reset(self):
    self.step_done = 0
    self.map = np.zeros((self.N,self.N))
    self.positions = self._compute_initial_positions(self.N, self.agent_ids)
    for agent_ids, position in self.positions.items():
      self.map[position[0]][position[1]] = agent_ids
    obs = self._get_obs()
    info = self._get_info()
    return (obs, info)


  def _check_if_action_possible(self, action):
    agent_id = action[0]
    move = action[1]
    position = self.positions[agent_id]
    new_position = [position[0] + move.value[0], position[1] + move.value[1]]

    r, c = new_position
    if not (0 <= r < self.N and 0 <= c < self.N):
        return False
    if self.map[r, c] == 0 or self.map[r, c] == agent_id:
      return True
    else:
      return False

  def _check_termination(self):
    cells_not_changed = np.sum(self.map==0)
    if cells_not_changed == 0:
      return True
    else:
      return False

  def _check_truncation(self):
    if self.step_done == self.max_step:
      return True
    else:
      return False

  def _get_obs(self):
    obs = {}
    for aid in self.agent_ids:
      mine = (self.map == aid).astype(np.float32)
      others = ((self.map != 0) & (self.map != aid)).astype(np.float32)
      empty = (self.map == 0).astype(np.float32)
      me = np.zeros((self.N, self.N), dtype=np.float32)
      r, c = self.positions[aid]
      me[r, c] = 1.0
      obs[aid] = np.stack([mine, others, empty, me], axis=0)  # (4, N, N)
    return obs

  def _get_info(self):
    cells = {aid: int(np.sum(self.map == aid)) for aid in self.agent_ids}
    return {"step_done": self.step_done, "cells": cells}

  def step(self, actions):
    self.step_done += 1
    random.shuffle(actions)
    for action in actions:
      if self._check_if_action_possible(action):
        agent_id = action[0]
        move = action[1]
        position = self.positions[agent_id]
        new_position = [position[0] + move.value[0], position[1] + move.value[1]]
        self.map[new_position[0]][new_position[1]] = agent_id
        self.positions[agent_id] = new_position
    obs = self._get_obs()
    terminated = self._check_termination()
    truncated = self._check_truncation()
    info = self._get_info()
    return obs, terminated, truncated, info

  def _compute_initial_positions(self, N, n_agent):
    if len(n_agent) == 2:
      positions = [[0,0],[N-1,N-1]]
      shuffle(positions)
      return {n_agent[0]: positions[0], n_agent[1]: positions[1]}
    elif len(n_agent) == 3:
      positions = [[0,0],[N-1,N-1],[N-1,0]]
      shuffle(positions)
      return {n_agent[0]: positions[0], n_agent[1]: positions[1], n_agent[2]: positions[2]}
    else:
      positions = [[0,0],[N-1,N-1],[N-1,0],[0,N-1]]
      shuffle(positions)
      return {n_agent[0]: positions[0], n_agent[1]: positions[1], n_agent[2]: positions[2], n_agent[3]: positions[3]}

  def render(self, cell_px=30):
    palette = {
      0: (235, 235, 235),
      1: (220, 60, 60),    # rosso
      2: (60, 120, 220),   # blu
      3: (60, 180, 90),    # verde
      4: (230, 180, 50),   # giallo
    }
    img = np.zeros((self.N, self.N, 3), dtype=np.uint8)
    for r in range(self.N):
      for c in range(self.N):
        img[r, c] = palette[int(self.map[r, c])]

    for aid, (r, c) in self.positions.items():
      img[r, c] = tuple(int(v * 0.55) for v in palette[aid])

    img = np.kron(img, np.ones((cell_px, cell_px, 1), dtype=np.uint8))
    return img
