import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

class EpisodeLoggerCallback(BaseCallback):
    """
    Custom callback for logging training and evaluation rewards 
    based on the episode number to meet the assignment requirements.
    """
    def __init__(self, eval_env, eval_freq_episodes=10, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq_episodes = eval_freq_episodes
        self.n_eval_episodes = n_eval_episodes
        
        # Lists to store data for plotting
        self.train_episodes = []
        self.train_rewards = []
        self.eval_episodes = []
        self.eval_rewards = []
        
        # Internal counters
        self.episode_count = 0
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        # Accumulate reward for the current timestep
        self.current_episode_reward += self.locals["rewards"][0]
        
        # Check if the episode is finished
        if self.locals["dones"][0]:
            self.episode_count += 1
            self.train_episodes.append(self.episode_count)
            self.train_rewards.append(self.current_episode_reward)
            
            # Reset the accumulated reward for the next episode
            self.current_episode_reward = 0.0
            
            # Perform evaluation at specified episode intervals
            if self.episode_count % self.eval_freq_episodes == 0:
                mean_reward, _ = evaluate_policy(
                    self.model, 
                    self.eval_env, 
                    n_eval_episodes=self.n_eval_episodes, 
                    deterministic=True
                )
                self.eval_episodes.append(self.episode_count)
                self.eval_rewards.append(mean_reward)
                
                if self.verbose > 0:
                    print(f"Episode {self.episode_count} - Eval Reward: {mean_reward:.2f}")
                    
        return True

def smooth_curve(scalars, weight=0.85):
    """
    Applies exponential moving average to smooth the training curve 
    for better visualization.
    """
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def run_q1a_baseline():
    """
    Question 1A: Train a standard PPO agent on continuous LunarLander-v3
    and generate the required performance plot.
    """
    print("--- Starting Question 1A: Baseline Training ---")
    
    # Initialize environments with continuous action space
    train_env = gym.make("LunarLander-v3", continuous=True)
    eval_env = gym.make("LunarLander-v3", continuous=True)
    
    # Setup callback
    logger_callback = EpisodeLoggerCallback(eval_env, eval_freq_episodes=10, n_eval_episodes=5, verbose=1)
    
    # Initialize PPO model
    model = PPO("MlpPolicy", train_env, verbose=0, seed=42)
    
    # Train the model (approx. 200,000 timesteps is usually enough for a basic landing)
    model.learn(total_timesteps=200000, callback=logger_callback)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot raw training curve (transparent) and smoothed training curve
    plt.plot(logger_callback.train_episodes, logger_callback.train_rewards, alpha=0.2, color='blue')
    smoothed_train = smooth_curve(logger_callback.train_rewards)
    plt.plot(logger_callback.train_episodes, smoothed_train, label='Curve 1: Training Curve (Smoothed)', color='blue')
    
    # Plot evaluation curve
    plt.plot(logger_callback.eval_episodes, logger_callback.eval_rewards, label='Curve 2: Evaluation Curve', color='red', linewidth=2)
    
    # Apply assignment formatting requirements
    plt.xlabel('Episode Number')
    plt.ylabel('Episode Reward')
    plt.title('PPO Performance on LunarLander-v3 (Continuous)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the figure
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/Q1A_Performance_Figure.png')
    plt.close()
    
    print("Baseline training completed. Plot saved to 'results/Q1A_Performance_Figure.png'.\n")

def run_q1b_tuning_experiments():
    """
    Question 1B: Run experiments with different hyperparameters to discuss 
    their effects on the agent's performance.
    """
    print("--- Starting Question 1B: Hyperparameter Tuning Experiments ---")
    
    # Define a dictionary of experiments to run
    experiments = {
        "Default": {},
        "High LR (0.01)": {"learning_rate": 0.01},
        "Low LR (0.0001)": {"learning_rate": 0.0001},
        "Low Gamma (0.90)": {"gamma": 0.90}, # Discount factor
        "Wider Network (256x256)": {"policy_kwargs": dict(net_arch=[256, 256])},
        "Low Clip Range (0.05)": {"clip_range": 0.05}
    }
    
    eval_results = {}
    
    # Use fewer timesteps for quick comparison
    total_timesteps = 100000 
    
    for exp_name, kwargs in experiments.items():
        print(f"Running experiment: {exp_name}...")
        train_env = gym.make("LunarLander-v3", continuous=True)
        eval_env = gym.make("LunarLander-v3", continuous=True)
        
        callback = EpisodeLoggerCallback(eval_env, eval_freq_episodes=15, n_eval_episodes=3, verbose=0)
        
        model = PPO("MlpPolicy", train_env, verbose=0, seed=42, **kwargs)
        model.learn(total_timesteps=total_timesteps, callback=callback)
        
        eval_results[exp_name] = (callback.eval_episodes, callback.eval_rewards)
    
    # Plot comparative evaluation curves
    plt.figure(figsize=(12, 8))
    for exp_name, (episodes, rewards) in eval_results.items():
        plt.plot(episodes, smooth_curve(rewards, weight=0.6), label=exp_name)
        
    plt.xlabel('Episode Number')
    plt.ylabel('Evaluation Episode Reward')
    plt.title('Hyperparameter Tuning Comparison (Evaluation Curves)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('results/Q1B_Tuning_Comparison.png')
    plt.close()
    
    print("Tuning experiments completed. Plot saved to 'results/Q1B_Tuning_Comparison.png'.")

if __name__ == "__main__":
    run_q1a_baseline()
    run_q1b_tuning_experiments()