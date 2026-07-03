from abc import ABC, abstractmethod

class RewardFunction(ABC):
  @abstractmethod
  def compute(self, prev_info, info, agent_id, terminated, truncated):
      pass

class DenseReward(RewardFunction):
  def compute(self, prev_info, info, agent_id, terminated, truncated):
    before = prev_info["cells"][agent_id]
    after = info["cells"][agent_id]
    return float(after - before)

class TerminalReward(RewardFunction):
    def compute(self, prev_info, info, agent_id, terminated, truncated):
        if not (terminated or truncated):
            return 0.0
        cells = info["cells"]
        mine = cells[agent_id]
        opp = max(v for k, v in cells.items() if k != agent_id)
        return float(mine - opp) #/ max(info["step_done"], 1)

class HybridReward(RewardFunction):
    def __init__(self, win_bonus=50.0):
      self.win_bonus = win_bonus

    def compute(self, prev_info, info, agent_id, terminated, truncated):
      before = prev_info["cells"][agent_id]
      after = info["cells"][agent_id]
      r = float(after - before)

      if terminated or truncated:
        cells = info["cells"]
        mine = cells[agent_id]
        opp = max(v for k, v in cells.items() if k != agent_id)
        r += self.win_bonus if mine > opp else -self.win_bonus

      return r
