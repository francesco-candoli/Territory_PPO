import numpy as np
import torch
from torch import nn
import torch.nn.functional as f
from torch.distributions import Categorical
from Buffer import Buffer

class PolicyNetwork(nn.Module):
  def __init__(self, N, in_channels, out_features, hidden_dim=128):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(),
    )
    conv_out = 64 * N * N
    self.head = nn.Sequential(
      nn.Linear(conv_out, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, out_features),
    )

  def forward(self, x):
    if x.dim() == 3:
      x = x.unsqueeze(0)
    x = self.conv(x)
    x = x.flatten(start_dim=1)
    x = self.head(x)
    return x

class PPOAgent:
  def __init__(self, obs_space, act_space, agent_id,
              lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
              value_coef=0.5, entropy_coef=0.01,
              epochs=4, minibatch_size=64, max_grad_norm=0.5,
              device="cpu"):
    self.agent_id = agent_id
    self.gamma = gamma
    self.lam = lam
    self.clip = clip
    self.value_coef = value_coef
    self.entropy_coef = entropy_coef
    self.epochs = epochs
    self.minibatch_size = minibatch_size
    self.max_grad_norm = max_grad_norm
    self.device = device
    self.in_channels = obs_space.shape[0]
    self.N      = obs_space.shape[1]
    self.n_actions   = act_space.n
    self.actor = PolicyNetwork(N=self.N, in_channels=self.in_channels, out_features=self.n_actions).to(device)
    self.critic = PolicyNetwork(N=self.N, in_channels=self.in_channels, out_features=1).to(device)
    self.optimizer = torch.optim.Adam(
      list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
    self.buffer = Buffer()

  @torch.no_grad()
  def act(self, obs, deterministic=False):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
    logits = self.actor(obs_t)          # (1, n_azioni)
    value = self.critic(obs_t)          # (1, 1)
    dist = Categorical(logits=logits)
    action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob.item(), value.item()

  def _compute_gae(self, last_value):
    rewards, dones = self.buffer.rewards, self.buffer.dones
    values = self.buffer.values + [last_value]
    advantages, gae = [], 0.0
    for t in reversed(range(len(rewards))):
      mask = 1.0 - float(dones[t])
      delta = rewards[t] + self.gamma * values[t+1] * mask - values[t]
      gae = delta + self.gamma * self.lam * mask * gae
      advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns

  def update(self, last_value=0.0):

    #PPO: 5. GAE to update
    advantages, returns = self._compute_gae(last_value)

    obs = torch.as_tensor(np.array(self.buffer.obs), dtype=torch.float32, device=self.device)
    actions = torch.as_tensor(self.buffer.actions, dtype=torch.long, device=self.device)
    old_log_probs = torch.as_tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

    #PPO: 6. +1e-8 to prevent the division by 0
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) 

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n": 0}

    n = len(self.buffer)
    idxs = np.arange(n)
    for _ in range(self.epochs):
      np.random.shuffle(idxs)
      for start in range(0, n, self.minibatch_size): #per ogni minibatch
        mb = idxs[start:start + self.minibatch_size]
        mb_obs, mb_act = obs[mb], actions[mb] #observations e actions prese
        mb_old_lp, mb_advantages, mb_returns = old_log_probs[mb], advantages[mb], returns[mb] #policy vecchia, vantaggi e ritorni

        dist = Categorical(logits=self.actor(mb_obs)) #distribution
        new_lp = dist.log_prob(mb_act) #new policy
        entropy = dist.entropy().mean() #mean entropy della distribuzione vecchia

        ratio = torch.exp(new_lp - mb_old_lp) #ratio between new policy and old policy
        surr1 = ratio * mb_advantages
        surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * mb_advantages 
        policy_loss = -torch.min(surr1, surr2).mean()

        values = self.critic(mb_obs).squeeze(-1)
        value_loss = f.mse_loss(values, mb_returns)

        stats["policy_loss"] += policy_loss.item()
        stats["value_loss"] += value_loss.item()
        stats["entropy"] += entropy.item()
        stats["n"] += 1

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
          list(self.actor.parameters()) + list(self.critic.parameters()),
          self.max_grad_norm)
        self.optimizer.step()

    self.buffer.clear()
    return {k: stats[k] / stats["n"] for k in ("policy_loss", "value_loss", "entropy")}

  def save(self, path):
    torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

  def load(self, path):
    weight = torch.load(path, map_location=self.device)
    self.actor.load_state_dict(weight["actor"])
    self.critic.load_state_dict(weight["critic"])
