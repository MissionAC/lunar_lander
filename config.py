# ==========================================
# Configuration settings for SB3 PPO Agent
# ==========================================

# Environment Settings
ENV_NAME = "LunarLander-v3"
IS_CONTINUOUS = True

# General Training Settings
TOTAL_TIMESTEPS = 500000
SEED = 42

# ------------------------------------------
# Custom Network Architecture Setup
# ------------------------------------------
# [64, 64] means 2 hidden layers with 64 neurons each.
NETWORK_ARCH = [64, 64] 

# PPO Hyperparameters (For Question 1B Tuning)
PPO_PARAMS = {
    "learning_rate": 0.0003,       # Learning rate
    "gamma": 0.99,                 # Discount factor
    "gae_lambda": 0.95,            # GAE trade-off
    "clip_range": 0.2,             # Clipping boundary
    "n_steps": 2048,               # Steps to run for each environment per update
    "batch_size": 64,              # Minibatch size
    "n_epochs": 10,                # Number of epoch when optimizing the surrogate loss
    "ent_coef": 0.0,               # Entropy coefficient for exploration
    
    # Injecting the custom network architecture into the policy
    "policy_kwargs": {
        "net_arch": dict(pi=NETWORK_ARCH, vf=NETWORK_ARCH)
    }
}

# Evaluation Settings
EVAL_FREQ_EPISODES = 20
N_EVAL_EPISODES = 10