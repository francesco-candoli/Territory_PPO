import random
import numpy as np
from Environment import Moves

class StaticAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

    def act(self, obs, deterministic=False):
        # obs: (4, N, N) -> canale 2 = vuote, canale 3 = la mia posizione
        empty = obs[2]
        me = obs[3]
        r, c = np.argwhere(me == 1.0)[0]      # la mia cella attuale
        N = obs.shape[1]

        empty_moves, own_moves = [], []
        for idx, move in enumerate(Moves):
            dr, dc = move.value
            nr, nc = r + dr, c + dc
            if not (0 <= nr < N and 0 <= nc < N):
                continue
            if empty[nr, nc] == 1.0:
                empty_moves.append(idx)
            else:
                if obs[0][nr, nc] == 1.0:
                    own_moves.append(idx)

        if empty_moves:
            return random.choice(empty_moves), None, None
        elif own_moves: 
            return random.choice(own_moves), None, None
        else:
            return 0, None, None
