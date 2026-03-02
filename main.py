import gymnasium as gym
from stable_baselines3 import PPO
import argparse
import os

# Import configurations and utility functions
import config
from utils import EpisodeLoggerCallback, plot_performance

def main():
    # Setup command line argument parsing for hyperparameter tuning
    parser = argparse.ArgumentParser(description="SB3 PPO Hyperparameter Tuning")
    parser.add_argument("--lr", type=float, default=config.PPO_PARAMS["learning_rate"], help="Learning rate")
    parser.add_argument("--gamma", type=float, default=config.PPO_PARAMS["gamma"], help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=config.PPO_PARAMS["gae_lambda"], help="GAE trade-off")
    parser.add_argument("--clip_range", type=float, default=config.PPO_PARAMS["clip_range"], help="Clipping boundary")
    parser.add_argument("--net_arch", type=int, nargs="+", default=config.NETWORK_ARCH, help="Network architecture (e.g., 64 64)")
    
    args = parser.parse_args()
    
    # Create a unique identifier string based on the current hyperparameters
    arch_str = "_".join(map(str, args.net_arch))
    param_str = f"lr{args.lr}_g{args.gamma}_gae{args.gae_lambda}_clip{args.clip_range}_arch{arch_str}"
    
    print(f"--- Starting SB3 PPO Training on {config.ENV_NAME} ---")
    print(f"Hyperparameters: {param_str}")
    
    # Update PPO parameters based on parsed arguments
    ppo_params = config.PPO_PARAMS.copy()
    ppo_params["learning_rate"] = args.lr
    ppo_params["gamma"] = args.gamma
    ppo_params["gae_lambda"] = args.gae_lambda
    ppo_params["clip_range"] = args.clip_range
    ppo_params["policy_kwargs"] = {
        "net_arch": dict(pi=args.net_arch, vf=args.net_arch)
    }
    
    # Initialize environments
    train_env = gym.make(config.ENV_NAME, continuous=config.IS_CONTINUOUS)
    eval_env = gym.make(config.ENV_NAME, continuous=config.IS_CONTINUOUS)
    
    # Setup custom logger callback for episode-based evaluation
    # Pass the param_str to the callback to save models properly
    logger_callback = EpisodeLoggerCallback(
        eval_env, 
        eval_freq_episodes=config.EVAL_FREQ_EPISODES, 
        n_eval_episodes=config.N_EVAL_EPISODES, 
        verbose=1,
        save_dir="results/models",
        param_str=param_str
    )
    
    # Initialize PPO model with updated parameters
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=0,
        device="auto",
        seed=config.SEED,
        **ppo_params 
    )
    
    # Train the agent
    print("Training in progress. Please wait...")
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=logger_callback)
    
    # Generate and save the performance plot locally, labeled with parameters
    plot_performance(logger_callback, param_str=param_str, save_dir='results/plots')
    
    print(f"--- Training Completed for {param_str} ---")

if __name__ == "__main__":
    main()