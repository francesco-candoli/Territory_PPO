class Buffer():
    def __init__(self):
        self.clear()

    def store(self, obs, action, log_prob, value, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.values, self.rewards, self.dones = [], [], []

    def __len__(self):
        return len(self.rewards)
