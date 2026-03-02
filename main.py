import gymnasium as gym
from stable_baselines3 import PPO

# Import configurations and utility functions
import config
from utils import EpisodeLoggerCallback, plot_performance

def main():
    print(f"--- Starting SB3 PPO Training on {config.ENV_NAME} ---")
    print(f"Network Architecture: {config.NETWORK_ARCH}")
    
    # Initialize environments
    train_env = gym.make(config.ENV_NAME, continuous=config.IS_CONTINUOUS)
    eval_env = gym.make(config.ENV_NAME, continuous=config.IS_CONTINUOUS)
    
    # Setup custom logger callback for episode-based evaluation
    logger_callback = EpisodeLoggerCallback(
        eval_env, 
        eval_freq_episodes=config.EVAL_FREQ_EPISODES, 
        n_eval_episodes=config.N_EVAL_EPISODES, 
        verbose=1
    )
    
    # Initialize PPO model with parameters from config.py
    # SB3 automatically detects and uses GPU if available
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=0,
        device="auto",
        seed=config.SEED,
        **config.PPO_PARAMS 
    )
    
    # Train the agent
    print("Training in progress. Please wait...")
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=logger_callback)
    
    # Generate and save the performance plot locally to submit for Question 1A
    plot_performance(logger_callback, save_path='results/SB3_Performance.png')
    
    print("--- Training Completed ---")

if __name__ == "__main__":
    main()