import os
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

class EpisodeLoggerCallback(BaseCallback):
    """
    Custom callback for logging training and evaluation rewards 
    based on the episode number to meet assignment requirements.
    """
    def __init__(self, eval_env, eval_freq_episodes=10, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq_episodes = eval_freq_episodes
        self.n_eval_episodes = n_eval_episodes
        
        # Data storage for local plotting
        self.train_episodes = []
        self.train_rewards = []
        self.eval_episodes = []
        self.eval_rewards = []
        
        # Internal tracking variables
        self.episode_count = 0
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        # Accumulate reward for the current timestep
        # locals["rewards"] is an array because SB3 wraps envs in a vectorized environment
        self.current_episode_reward += self.locals["rewards"][0]
        
        # Check if the episode is finished
        if self.locals["dones"][0]:
            self.episode_count += 1
            self.train_episodes.append(self.episode_count)
            self.train_rewards.append(self.current_episode_reward)
            
            # Reset accumulated reward
            self.current_episode_reward = 0.0
            
            # Evaluate agent periodically based on episode count
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
                    print(f"Episode {self.episode_count} | Eval Reward: {mean_reward:.2f}")
                    
        return True

def smooth_curve(scalars, weight=0.85):
    """
    Applies exponential moving average to smooth curves.
    """
    if not scalars:
        return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_performance(callback_data, save_path='results/SB3_Performance_Figure.png'):
    """
    Generates and saves the performance plot required by the assignment.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot training curves
    plt.plot(callback_data.train_episodes, callback_data.train_rewards, alpha=0.2, color='blue')
    smoothed_train = smooth_curve(callback_data.train_rewards)
    plt.plot(callback_data.train_episodes, smoothed_train, label='Curve 1: Training Curve (Smoothed)', color='blue')
    
    # Plot evaluation curve
    plt.plot(callback_data.eval_episodes, callback_data.eval_rewards, label='Curve 2: Evaluation Curve', color='red', linewidth=2, marker='o')
    
    # Formatting
    plt.xlabel('Episode Number')
    plt.ylabel('Episode Reward')
    plt.title('SB3 PPO Performance on LunarLander-v3 (Continuous)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"\nPlot successfully saved to {save_path}")