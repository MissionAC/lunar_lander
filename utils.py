import os
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

class EpisodeLoggerCallback(BaseCallback):
    """
    Custom callback for logging training and evaluation rewards.
    Modified to track and save the top 3 models based on evaluation performance.
    """
    def __init__(self, eval_env, eval_freq_episodes=10, n_eval_episodes=5, verbose=0, save_dir="results/models", param_str="default"):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq_episodes = eval_freq_episodes
        self.n_eval_episodes = n_eval_episodes
        
        # Configuration for saving models
        self.save_dir = save_dir
        self.param_str = param_str
        os.makedirs(self.save_dir, exist_ok=True)
        
        # List to keep track of the top 3 models. Format: [(mean_reward, filepath), ...]
        self.best_models = []
        
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
                
                # Logic to save the top 3 models
                # If we have less than 3 models, or the current reward is better than the worst in our top 3
                if len(self.best_models) < 3 or mean_reward > self.best_models[0][0]:
                    model_filename = f"model_{self.param_str}_ep{self.episode_count}_rew{mean_reward:.2f}.zip"
                    model_path = os.path.join(self.save_dir, model_filename)
                    
                    # Save the new top model
                    self.model.save(model_path)
                    
                    # Add to our tracking list and sort by reward (ascending)
                    self.best_models.append((mean_reward, model_path))
                    self.best_models.sort(key=lambda x: x[0])
                    
                    if self.verbose > 0:
                        print(f"--> Saved new top model: {model_filename}")
                    
                    # If we exceeded 3 models, remove the worst one
                    if len(self.best_models) > 3:
                        worst_reward, worst_path = self.best_models.pop(0)
                        if os.path.exists(worst_path):
                            os.remove(worst_path)
                            if self.verbose > 0:
                                print(f"--> Removed lower performing model: {os.path.basename(worst_path)}")
                    
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

def plot_performance(callback_data, param_str="default", save_dir='results/plots'):
    """
    Generates and saves the performance plot required by the assignment.
    The filename and title will include the hyperparameter string.
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
    
    # Dynamic title based on parameters
    plt.title(f'SB3 PPO Performance\nParams: {param_str}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the figure with parameter string in the filename
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'Performance_{param_str}.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"\nPlot successfully saved to {save_path}")