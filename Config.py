import os

from Rewards import DenseReward, HybridReward, TerminalReward

# --- Environment ---
N = 15
N_AGENT = 2

def default_max_step(n):
    return n * n * 3
MAX_STEP = default_max_step(N)  # = 675

# --- PPO hyperparameters (PPOAgent / Trainer defaults) ---
LEARNING_RATE = 3e-4
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
EPOCHS = 4
MINIBATCH_SIZE = 64
MAX_GRAD_NORM = 0.5
ROLLOUT_LEN = 2048
GAMMA_BY_VARIANT = {
    "Dense": 0.99,
    "Terminal": 0.999,
    "Hybrid": 0.99,
}

# --- Rewards ---
WIN_BONUS = 50.0
REWARD_REGISTRY = {
    "Dense": DenseReward,
    "Terminal": TerminalReward,
    "Hybrid": HybridReward,
}

# --- Training defaults ---
DEFAULT_TRAIN_STEPS = 2_000_000

# --- Evaluation / tournament defaults ---
DEFAULT_GAMES = 100
TOURNAMENT_SEEDS = (0, 44, 999)
DEFAULT_FPS = 20

# --- Paths ---
DATA_DIR = "data"
MODELS_DIR = os.path.join(DATA_DIR, "models")
HISTORY_DIR = os.path.join(DATA_DIR, "history")
GIF_DIR = os.path.join(DATA_DIR, "gif")
TOURNAMENT_RESULTS_PATH = os.path.join(DATA_DIR, "tournament_results.json")

def weight_filename(n, variant, steps):
    return f"weight_N{n}_{variant}_{steps}.pt"

def history_filename(n, variant, steps):
    return f"history_N{n}_{variant}_{steps}.json"

DEFAULT_MODEL_PATHS = {
    variant: os.path.join(MODELS_DIR, weight_filename(N, variant, DEFAULT_TRAIN_STEPS))
    for variant in ("Dense", "Terminal", "Hybrid")
}
